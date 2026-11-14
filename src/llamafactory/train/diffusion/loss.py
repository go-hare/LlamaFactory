"""Section 4 conversion step for a causal Gemma checkpoint.

The report's conversion SFT (equation 11) is decoder cross-entropy only.
A canvas is 256 tokens. Tokens are replaced by uniform vocabulary ids with
probability t ~ U(0, 1). The decoder attends to the clean prompt and earlier
canvases, and bidirectionally inside the selected canvas. It does not see
later canvases. There is no encoder next-token loss. Half the rows feed a
stopped first pass back through the self-conditioning block; the rest use a
zero signal. The report states that 50 percent split in the downstream
recipe. Equation 11 is written as a sum. The released loss class is not in this checkout, so this stays a mean over valid canvas positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _chunk(response: torch.Tensor, num_canvases: int, canvas_size: int, eos_id: int, pad_id: int):
    total = num_canvases * canvas_size
    flat = torch.full((total,), pad_id, dtype=response.dtype, device=response.device)
    n = min(int(response.numel()), total)
    if n:
        flat[:n] = response[:n]
        last = (n - 1) // canvas_size
        flat[n : (last + 1) * canvas_size] = eos_id
        n_valid = last + 1
    else:
        n_valid = 0
    canvas_id = torch.arange(num_canvases, device=response.device).repeat_interleave(canvas_size)
    mask = torch.zeros(total, dtype=torch.bool, device=response.device)
    mask[: n_valid * canvas_size] = True
    return flat, canvas_id, mask


def _allow_mask(
    prompt_valid: torch.Tensor,
    canvas_valid: torch.Tensor,
    *,
    canvas_bidirectional: bool,
) -> torch.Tensor:
    """True means attend. Prompt is causal and cannot see the canvas.

    Canvas queries see the real prompt. Training makes the valid canvas
    bidirectional; the encoder keeps it causal so pads after the response stay hidden.
    """
    batch, prompt_len = prompt_valid.shape
    canvas_len = canvas_valid.shape[1]
    device = prompt_valid.device
    query = torch.arange(prompt_len, device=device)[:, None]
    key = torch.arange(prompt_len, device=device)[None, :]
    prompt_prompt = (query >= key)[None, :, :] & prompt_valid[:, None, :]
    prompt_canvas = torch.zeros(batch, prompt_len, canvas_len, dtype=torch.bool, device=device)
    canvas_prompt = prompt_valid[:, None, :] & canvas_valid[:, :, None]
    if canvas_bidirectional:
        canvas_canvas = canvas_valid[:, None, :] & canvas_valid[:, :, None]
    else:
        canvas_query = torch.arange(canvas_len, device=device)[:, None]
        canvas_key = torch.arange(canvas_len, device=device)[None, :]
        canvas_canvas = (canvas_query >= canvas_key)[None, :, :] & canvas_valid[:, None, :]
    top = torch.cat([prompt_prompt, prompt_canvas], dim=2)
    bottom = torch.cat([canvas_prompt, canvas_canvas], dim=2)
    return torch.cat([top, bottom], dim=1)[:, None, :, :]


class SelfConditioner(nn.Module):
    """Official ``SelfConditioning``: RMSNorm, gated FeedForward (GeGLU), add, RMSNorm.

    The signal is the previous canvas distribution embedded with the token table.
    A zero signal stays zero and does not change the canvas embeddings.
    """

    def __init__(self, features: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = nn.RMSNorm(features, eps=eps)
        self.gate_proj = nn.Linear(features, hidden_dim, bias=False)
        self.up_proj = nn.Linear(features, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, features, bias=False)
        self.post_norm = nn.RMSNorm(features, eps=eps, elementwise_affine=False)

    def forward(self, canvas_embeddings: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        # A zero signal still passes the post-norm, matching the official block.
        if torch.count_nonzero(signal) == 0:
            added = torch.zeros_like(canvas_embeddings)
        else:
            normed = self.pre_norm(signal)
            added = self.down_proj(F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        return self.post_norm(canvas_embeddings + added)


def _token_embedding(model) -> nn.Module:
    """The table used both to look up tokens and to embed a stopped distribution.

    Factory loads ``Gemma4ForConditionalGeneration``, whose table is
    ``language_model.embed_tokens``. A causal wrapper keeps it on ``model``.
    Peft wraps either one in ``base_model.model``.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    language = getattr(getattr(base, "model", None), "language_model", None)
    if language is not None and hasattr(language, "embed_tokens"):
        return language.embed_tokens
    text = getattr(base, "model", None)
    if text is not None and hasattr(text, "embed_tokens"):
        return text.embed_tokens
    embedding = base.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise RuntimeError("Diffusion self-conditioning cannot find the token embedding.")
    return embedding


def _embed_distribution(token_embedding: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """``Embedder.encode_logits``: softmax over the vocabulary, times E, times sqrt(d)."""
    weights = torch.softmax(logits.float(), dim=-1).to(token_embedding.weight.dtype)
    scale = float(token_embedding.weight.shape[-1]) ** 0.5
    return (weights @ token_embedding.weight) * scale


def _get_conditioner(model) -> SelfConditioner:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for holder in (model, base, getattr(base, "model", None)):
        if holder is not None and hasattr(holder, "self_conditioner"):
            cond = getattr(holder, "self_conditioner")
            if cond is not None:
                return cond
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    intermediate_size = int(getattr(text_config, "intermediate_size", 0) or hidden_size * 4)
    emb = _token_embedding(model).weight
    conditioner = SelfConditioner(hidden_size, intermediate_size).to(device=emb.device, dtype=emb.dtype)
    model.self_conditioner = conditioner
    return conditioner


def _additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _gemma_masks(
    allowed: torch.Tensor,
    dtype: torch.dtype,
    *,
    valid: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> dict[str, torch.Tensor]:
    """Gemma 4 uses a mask dict as-is, so the sliding layer needs its own band.

    Positions follow the valid tokens. Right padding must not push the canvas
    outside the window.
    """
    full = _additive(allowed, dtype)
    if sliding_window is None or valid is None:
        return {"full_attention": full, "sliding_attention": full}
    positions = valid.long().cumsum(dim=-1) - 1
    positions = positions.clamp(min=0)
    distance = (positions[:, :, None] - positions[:, None, :]).abs()
    band = distance < sliding_window
    while band.ndim < allowed.ndim:
        band = band[:, None]
    return {"full_attention": full, "sliding_attention": _additive(allowed & band, dtype)}


def diffusion_batch_step(
    model,
    prompts: list[torch.Tensor],
    responses: list[torch.Tensor],
    *,
    canvas_size: int,
    num_canvases: int,
    pad_id: int,
    eos_id: int,
    time_eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched section-4 conversion step across ``B`` rows in two forward passes."""
    if not prompts:
        raise ValueError("Diffusion batch is empty.")
    device = _token_embedding(model).weight.device
    text_config = getattr(model.config, "text_config", model.config)
    vocab_size = int(getattr(text_config, "vocab_size", 0) or model.config.vocab_size)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    window = getattr(text_config, "sliding_window", None)
    conditioner = _get_conditioner(model)
    token_embedding = _token_embedding(model)
    model_dtype = token_embedding.weight.dtype

    batch_size = len(prompts)
    history_seqs: list[torch.Tensor] = []
    history_valids: list[torch.Tensor] = []
    current_noisy: list[torch.Tensor] = []
    current_targets: list[torch.Tensor] = []
    current_valids: list[torch.Tensor] = []

    first_indices = torch.arange(num_canvases, device=device) * canvas_size
    for p_raw, r_raw in zip(prompts, responses):
        p = p_raw.to(device=device, dtype=torch.long).view(-1)
        r = r_raw.to(device=device, dtype=torch.long).view(-1)
        if p.numel() == 0:
            raise ValueError("Diffusion prompt is empty.")
        canvas, _, canvas_mask = _chunk(r, num_canvases, canvas_size, eos_id, pad_id)
        t = torch.empty((), device=device).uniform_(time_eps, 1.0 - time_eps)
        noise = torch.randint(0, vocab_size, canvas.shape, device=device)
        move = torch.rand(canvas.shape, device=device) < t
        xt = torch.where(move, noise, canvas)

        num_valid = canvas_mask[first_indices].sum().clamp(min=1)
        selected_i = int((torch.rand((), device=device) * num_valid).long().item())
        prefix_len = selected_i * canvas_size

        hist = torch.cat([p, canvas[:prefix_len]], dim=0)
        hist_valid = torch.cat([p != pad_id, canvas_mask[:prefix_len]], dim=0)
        history_seqs.append(hist)
        history_valids.append(hist_valid)
        current_noisy.append(xt[prefix_len : prefix_len + canvas_size])
        current_targets.append(canvas[prefix_len : prefix_len + canvas_size])
        current_valids.append(canvas_mask[prefix_len : prefix_len + canvas_size])

    max_hist = max(h.numel() for h in history_seqs)
    decoder_tokens = torch.full((batch_size, max_hist + canvas_size), pad_id, dtype=torch.long, device=device)
    history_valid_batch = torch.zeros((batch_size, max_hist), dtype=torch.bool, device=device)
    for b in range(batch_size):
        hlen = history_seqs[b].numel()
        decoder_tokens[b, :hlen] = history_seqs[b]
        decoder_tokens[b, max_hist:] = current_noisy[b]
        history_valid_batch[b, :hlen] = history_valids[b]

    current_valid_batch = torch.stack(current_valids, dim=0)
    target_batch = torch.stack(current_targets, dim=0)
    valid_batch = torch.cat([history_valid_batch, current_valid_batch], dim=1)
    position_ids = (valid_batch.long().cumsum(dim=-1) - 1).clamp(min=0)

    total = max_hist + canvas_size
    allowed = torch.zeros((batch_size, 1, total, total), dtype=torch.bool, device=device)
    hist_causal = _allow_mask(history_valid_batch, current_valid_batch[:, :0], canvas_bidirectional=False)
    allowed[:, :, :max_hist, :max_hist] = hist_causal
    full_canvas_allow = _allow_mask(history_valid_batch, current_valid_batch, canvas_bidirectional=True)
    allowed[:, :, max_hist:, :] = full_canvas_allow[:, :, max_hist:, :]
    attn_masks = _gemma_masks(allowed, model_dtype, valid=valid_batch, sliding_window=window)

    def decode(signal: torch.Tensor):
        original = token_embedding.forward

        def embed(input_ids):
            embeddings = original(input_ids)
            if embeddings.shape[1] == total and signal.shape[-1] == embeddings.shape[-1]:
                embeddings = embeddings.clone()
                for b in range(embeddings.shape[0]):
                    embeddings[b : b + 1, max_hist:] = conditioner(
                        embeddings[b : b + 1, max_hist:], signal[b : b + 1]
                    )
            return embeddings

        token_embedding.forward = embed
        try:
            return model(
                input_ids=decoder_tokens,
                attention_mask=attn_masks,
                position_ids=position_ids,
                use_cache=False,
            )
        finally:
            token_embedding.forward = original

    zero_signal = torch.zeros((batch_size, canvas_size, hidden_size), device=device, dtype=model_dtype)
    with torch.no_grad():
        first = decode(zero_signal)
    first_logits = first.logits[:, max_hist:]
    embedded_first = _embed_distribution(token_embedding, first_logits)
    use_signal = (torch.rand((batch_size, 1, 1), device=device) < 0.5)
    signal = torch.where(use_signal, embedded_first, zero_signal)

    decoder = decode(signal)
    current_logits = decoder.logits[:, max_hist:]

    row_losses: list[torch.Tensor] = []
    for b in range(batch_size):
        c_valid = current_valid_batch[b]
        if not bool(c_valid.any()):
            raise RuntimeError("Diffusion decoder selected no canvas tokens.")
        sel_logits = current_logits[b][c_valid]
        if not torch.isfinite(sel_logits).all():
            raise RuntimeError("Decoder logits are non-finite on the selected canvas.")
        row_losses.append(F.cross_entropy(sel_logits, target_batch[b][c_valid]))
    mean_loss = torch.stack(row_losses).mean()
    return mean_loss, {"decoder_loss": float(mean_loss.detach())}


def diffusion_step(
    model,
    prompt: torch.Tensor,
    response: torch.Tensor,
    *,
    canvas_size: int,
    num_canvases: int,
    pad_id: int,
    eos_id: int,
    time_eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One training row. ``response`` is the unpadded canvas token ids."""
    return diffusion_batch_step(
        model,
        [prompt],
        [response],
        canvas_size=canvas_size,
        num_canvases=num_canvases,
        pad_id=pad_id,
        eos_id=eos_id,
        time_eps=time_eps,
    )


def denoise_step_logits(
    model,
    prompt: torch.Tensor,
    canvas: torch.Tensor,
    *,
    sc_embeddings: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Single decoder forward pass with carried self-conditioning embeddings ``sc_embeddings``."""
    prompt = prompt.view(1, -1)
    canvas = canvas.view(1, -1)
    if prompt.shape[1] == 0:
        raise ValueError("Diffusion prompt is empty.")
    text_config = getattr(model.config, "text_config", model.config)
    window = getattr(text_config, "sliding_window", None)
    prompt_valid = prompt != pad_id
    canvas_valid = torch.ones_like(canvas, dtype=torch.bool)
    history_len = prompt.shape[1]
    total = history_len + canvas.shape[1]
    allowed = prompt.new_zeros(1, 1, total, total, dtype=torch.bool)
    allowed[:, :, :history_len, :history_len] = torch.tril(prompt_valid[:, None, :].expand(-1, history_len, -1))
    canvas_rows = _allow_mask(prompt_valid, canvas_valid, canvas_bidirectional=True)
    allowed[:, :, history_len:, :] = canvas_rows[:, :, history_len:, :]
    tokens = torch.cat([prompt, canvas], dim=1)
    valid = torch.cat([prompt_valid, canvas_valid], dim=1)
    conditioner = _get_conditioner(model)
    token_embedding = _token_embedding(model)
    model_dtype = token_embedding.weight.dtype
    signal = sc_embeddings.to(device=tokens.device, dtype=model_dtype)
    original = token_embedding.forward

    def embed(input_ids):
        embeddings = original(input_ids)
        if embeddings.shape[1] == tokens.shape[1] and signal.shape[-1] == embeddings.shape[-1]:
            block = embeddings[:, history_len:]
            embeddings = embeddings.clone()
            embeddings[:, history_len:] = conditioner(block, signal)
        return embeddings

    token_embedding.forward = embed
    try:
        out = model(
            input_ids=tokens,
            attention_mask=_gemma_masks(allowed, model_dtype, valid=valid, sliding_window=window),
            use_cache=False,
        )
    finally:
        token_embedding.forward = original
    return out.logits[:, history_len:]


def denoise_logits(model, prompt: torch.Tensor, canvas: torch.Tensor, *, pad_id: int) -> torch.Tensor:
    """Two-pass convenience wrapper when no prior ``sc_embeddings`` are carried."""
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    canvas_len = canvas.numel()
    zero_sc = torch.zeros(1, canvas_len, hidden_size, device=prompt.device, dtype=_token_embedding(model).weight.dtype)
    with torch.no_grad():
        first_logits = denoise_step_logits(model, prompt, canvas, sc_embeddings=zero_sc, pad_id=pad_id)
    signal = _embed_distribution(_token_embedding(model), first_logits)
    return denoise_step_logits(model, prompt, canvas, sc_embeddings=signal, pad_id=pad_id)
