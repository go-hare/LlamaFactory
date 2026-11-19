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

import inspect
import math

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import _additive_mask, _token_embedding, get_diffusion_adapter
from .objective import (
    CorruptionProcess,
    DiffusionObjective,
    resolve_corruption,
    resolve_objective,
)


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
    A zero signal adds no feed-forward signal; post RMSNorm still applies.
    """

    def __init__(self, features: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = nn.RMSNorm(features, eps=eps)
        self.gate_proj = nn.Linear(features, hidden_dim, bias=False)
        self.up_proj = nn.Linear(features, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, features, bias=False)
        self.post_norm = nn.RMSNorm(features, eps=eps, elementwise_affine=False)
        # Official Gemma4 FeedForward uses Einsum's normal(stddev=0.01).
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)

    def forward(self, canvas_embeddings: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(signal)
        added = self.down_proj(F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        return self.post_norm(canvas_embeddings + added)


def _token_embedding(model) -> nn.Module:
    """Delegate to ``adapter._token_embedding`` for backward compatibility."""
    from .adapter import _token_embedding as _impl

    return _impl(model)


def _embed_distribution(token_embedding: nn.Module, logits: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
    """``Embedder.encode_logits``: softmax over the vocabulary, times E, times adapter embed scale."""
    weights = torch.softmax(logits.float(), dim=-1).to(token_embedding.weight.dtype)
    if scale is None:
        scale = float(token_embedding.weight.shape[-1]) ** 0.5
    return (weights @ token_embedding.weight) * scale


def rcd_residual_signal(
    token_embedding: nn.Module,
    logits: torch.Tensor,
    canvas_tokens: torch.Tensor,
    *,
    mask_id: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """RCD entropy-gated contextual residual: ``alpha_i * (p_i @ E)`` on masked positions.

    ``alpha_i = 1 - H(p_i) / log(|V|)`` downweights high-entropy noise and zeros out
    already-unmasked positions so only uncertain/remasked tokens carry residuals.
    """
    shaped = logits.float()
    vocab_size = shaped.shape[-1]
    if mask_id is not None and 0 <= int(mask_id) < vocab_size:
        shaped = shaped.clone()
        shaped[:, :, int(mask_id)] = -float("inf")
        eff_vocab = max(vocab_size - 1, 2)
    else:
        eff_vocab = max(vocab_size, 2)
    log_probs = torch.log_softmax(shaped, dim=-1)
    probs = torch.exp(log_probs)
    safe_log = torch.where(probs == 0, torch.zeros_like(log_probs), log_probs)
    entropy = -(safe_log * probs).sum(dim=-1)
    alpha = (1.0 - entropy / math.log(eff_vocab)).clamp(0.0, 1.0)
    if scale is None:
        scale = float(token_embedding.weight.shape[-1]) ** 0.5
    expected = (probs.to(token_embedding.weight.dtype) @ token_embedding.weight) * scale
    weighted = expected * alpha.unsqueeze(-1).to(token_embedding.weight.dtype)
    if mask_id is not None:
        is_masked = (canvas_tokens == int(mask_id)).unsqueeze(-1)
        return torch.where(is_masked, weighted, torch.zeros_like(weighted))
    return weighted


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


def prepare_conversion_training(model):
    get_diffusion_adapter(model).prepare_training(model)
    _get_conditioner(model)


def _supports_logits_to_keep(model) -> bool:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cached = getattr(base, "_diffusion_supports_logits_to_keep", None)
    if cached is not None:
        return bool(cached)
    try:
        supported = "logits_to_keep" in inspect.signature(base.forward).parameters
    except (TypeError, ValueError):
        supported = False
    try:
        setattr(base, "_diffusion_supports_logits_to_keep", supported)
    except Exception:
        pass
    return supported


def _diffusion_forward(
    model,
    tokens,
    signal,
    canvas_start,
    attention_mask,
    position_ids,
    *,
    logits_to_keep: int | None = None,
):
    """Delegate model-specific embedding/PLE/hook behavior to the resolved ``DiffusionModelAdapter``."""
    conditioner = _get_conditioner(model)
    adapter = get_diffusion_adapter(model)
    extra = {"logits_to_keep": int(logits_to_keep)} if (logits_to_keep is not None and _supports_logits_to_keep(model)) else {}
    with adapter.forward_context(model, tokens, signal, canvas_start, conditioner):
        return model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **extra,
        )


def _additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _additive_mask(allowed, dtype)


def _gemma_masks(
    allowed: torch.Tensor,
    dtype: torch.dtype,
    *,
    valid: torch.Tensor | None = None,
    sliding_window: int | None = None,
    model=None,
) -> dict[str, torch.Tensor]:
    """Build attention mask mapping via the model's ``DiffusionModelAdapter``."""
    if model is not None:
        return get_diffusion_adapter(model).build_attention_mask(
            allowed, dtype, valid=valid, sliding_window=sliding_window
        )
    full = _additive(allowed, dtype)
    if sliding_window is None or valid is None:
        return {"full_attention": full, "sliding_attention": full}
    positions = (valid.long().cumsum(dim=-1) - 1).clamp(min=0)
    distance = (positions[:, :, None] - positions[:, None, :]).abs()
    band = distance < sliding_window
    while band.ndim < allowed.ndim:
        band = band[:, None]
    return {"full_attention": full, "sliding_attention": _additive(allowed & band, dtype)}


def canvas_objective(logits, targets, valid, body_tokens, mode='canvas-mean', *, corrupted=None, time: float = 1.0):
    """Delegate canvas objective evaluation to ``objective.resolve_objective``."""
    objective = resolve_objective(mode)
    if corrupted is None:
        corrupted = torch.ones_like(valid, dtype=torch.bool)
    return objective.compute(logits, targets, valid, corrupted, body_tokens=body_tokens, time=time)


def _extract_canvas_logits(
    full_logits: torch.Tensor,
    history_lengths: list[int],
    max_hist: int,
    canvas_size: int,
    *,
    token_shift: bool,
) -> torch.Tensor:
    """Slice per-row canvas logits; ``token_shift=True`` reads position i-1 to predict i (Fast-dLLM v2)."""
    if not token_shift:
        return full_logits[:, max_hist : max_hist + canvas_size]
    if canvas_size == 1:
        return torch.stack([full_logits[b, hlen - 1 : hlen] for b, hlen in enumerate(history_lengths)], dim=0)
    rows = []
    tail = full_logits[:, max_hist : max_hist + canvas_size - 1]
    for b, hlen in enumerate(history_lengths):
        head = full_logits[b, hlen - 1 : hlen]
        rows.append(torch.cat([head, tail[b]], dim=0))
    return torch.stack(rows, dim=0)


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
    time_min: float | None = None,
    time_max: float | None = None,
    diagnostics: bool = False,
    loss_mode: str | DiffusionObjective = 'canvas-mean',
    corruption_mode: str | CorruptionProcess = 'uniform',
    mask_id: int | None = None,
    token_shift: bool = False,
    complementary_mask: bool = False,
    residual_mode: str = "standard",
    ref_model=None,
) -> tuple[torch.Tensor, dict]:
    """Batched section-4 conversion step across ``B`` rows in two forward passes."""
    if not prompts:
        raise ValueError("Diffusion batch is empty.")
    if residual_mode not in ("standard", "rcd"):
        raise ValueError(f"Unknown residual_mode: {residual_mode}")
    corruption = resolve_corruption(corruption_mode, mask_id=mask_id)
    objective = resolve_objective(loss_mode)
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
    current_corrupted: list[torch.Tensor] = []
    sample_times: list[float] = []
    diagnostic_rows = []
    body_lengths = []

    first_indices = torch.arange(num_canvases, device=device) * canvas_size
    for p_raw, r_raw in zip(prompts, responses):
        p = p_raw.to(device=device, dtype=torch.long).view(-1)
        r = r_raw.to(device=device, dtype=torch.long).view(-1)
        if p.numel() == 0:
            raise ValueError("Diffusion prompt is empty.")
        canvas, _, canvas_mask = _chunk(r, num_canvases, canvas_size, eos_id, pad_id)
        corrupted_sample = corruption.corrupt(
            canvas,
            vocab_size=vocab_size,
            time_eps=time_eps,
            time_min=time_min,
            time_max=time_max,
        )
        xt = corrupted_sample.noisy
        move = corrupted_sample.corrupted
        t = corrupted_sample.time

        num_valid = canvas_mask[first_indices].sum().clamp(min=1)
        selected_i = int((torch.rand((), device=device) * num_valid).long().item())
        prefix_len = selected_i * canvas_size
        body_lengths.append(max(0, min(canvas_size, r.numel() - prefix_len)))
        if diagnostics:
            diagnostic_rows.append(dict(
                time=t, selected_block=selected_i,
                body_tokens=max(0, min(canvas_size, r.numel() - prefix_len)),
                changed_fraction=float(move[prefix_len:prefix_len + canvas_size].float().mean()),
            ))

        hist = torch.cat([p, canvas[:prefix_len]], dim=0)
        hist_valid = torch.cat([p != pad_id, canvas_mask[:prefix_len]], dim=0)
        history_seqs.append(hist)
        history_valids.append(hist_valid)
        current_noisy.append(xt[prefix_len : prefix_len + canvas_size])
        current_targets.append(canvas[prefix_len : prefix_len + canvas_size])
        current_valids.append(canvas_mask[prefix_len : prefix_len + canvas_size])
        current_corrupted.append(move[prefix_len : prefix_len + canvas_size])
        sample_times.append(t)

    max_hist = max(h.numel() for h in history_seqs)
    history_lengths = [h.numel() for h in history_seqs]
    decoder_tokens = torch.full((batch_size, max_hist + canvas_size), pad_id, dtype=torch.long, device=device)
    history_valid_batch = torch.zeros((batch_size, max_hist), dtype=torch.bool, device=device)
    for b in range(batch_size):
        hlen = history_lengths[b]
        decoder_tokens[b, :hlen] = history_seqs[b]
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
    attn_masks = _gemma_masks(allowed, model_dtype, valid=valid_batch, sliding_window=window, model=model)

    adapter = get_diffusion_adapter(model)
    zero_signal = torch.zeros((batch_size, canvas_size, hidden_size), device=device, dtype=model_dtype)

    def _run_view_loss(
        view_noisy: list[torch.Tensor],
        view_corrupted: list[torch.Tensor],
        view_times: list[float],
        *,
        record_diagnostics: bool,
    ) -> torch.Tensor:
        view_tokens = decoder_tokens.clone()
        for b in range(batch_size):
            view_tokens[b, max_hist:] = view_noisy[b]
        ref = ref_model if (residual_mode == "rcd" and ref_model is not None) else model
        ref_keep = canvas_size if (not token_shift and _supports_logits_to_keep(ref)) else None
        use_signal = torch.rand((batch_size, 1, 1), device=device) < 0.5
        signal = zero_signal
        if bool(use_signal.any()):
            active_idx = use_signal.view(-1).nonzero(as_tuple=False).flatten()
            if active_idx.numel() == batch_size:
                sub_tokens = view_tokens
                sub_zero = zero_signal
                sub_masks = attn_masks
                sub_pos = position_ids
                sub_hist_lens = history_lengths
            else:
                sub_tokens = view_tokens.index_select(0, active_idx)
                sub_zero = zero_signal.index_select(0, active_idx)
                sub_masks = {k: v.index_select(0, active_idx) for k, v in attn_masks.items()}
                sub_pos = position_ids.index_select(0, active_idx)
                sub_hist_lens = [history_lengths[int(i)] for i in active_idx.tolist()]
            with torch.no_grad():
                first = _diffusion_forward(
                    ref, sub_tokens, sub_zero, max_hist, sub_masks, sub_pos, logits_to_keep=ref_keep
                )
            first_logits = (
                first.logits
                if ref_keep is not None
                else _extract_canvas_logits(first.logits, sub_hist_lens, max_hist, canvas_size, token_shift=token_shift)
            )
            if residual_mode == "rcd":
                embedded_first = rcd_residual_signal(
                    token_embedding,
                    first_logits,
                    sub_tokens[:, max_hist:],
                    mask_id=mask_id,
                    scale=adapter.embed_scale(model),
                )
            else:
                embedded_first = _embed_distribution(token_embedding, first_logits, scale=adapter.embed_scale(model))
            if active_idx.numel() == batch_size:
                signal = embedded_first
            else:
                signal = zero_signal.clone()
                signal.index_copy_(0, active_idx, embedded_first)

        dec_keep = canvas_size if (not token_shift and _supports_logits_to_keep(model)) else None
        decoder = _diffusion_forward(
            model, view_tokens, signal, max_hist, attn_masks, position_ids, logits_to_keep=dec_keep
        )
        current_logits = (
            decoder.logits
            if dec_keep is not None
            else _extract_canvas_logits(decoder.logits, history_lengths, max_hist, canvas_size, token_shift=token_shift)
        )

        row_losses: list[torch.Tensor] = []
        for b in range(batch_size):
            c_valid = current_valid_batch[b]
            if not bool(c_valid.any()):
                raise RuntimeError("Diffusion decoder selected no canvas tokens.")
            sel_logits = current_logits[b][c_valid]
            if not torch.isfinite(sel_logits).all():
                raise RuntimeError("Decoder logits are non-finite on the selected canvas.")
            row_losses.append(
                objective.compute(
                    current_logits[b],
                    target_batch[b],
                    c_valid,
                    view_corrupted[b],
                    body_tokens=body_lengths[b],
                    time=view_times[b],
                )
            )
            if record_diagnostics:
                meta = diagnostic_rows[b]
                ce = F.cross_entropy(current_logits[b].detach().float(), target_batch[b], reduction='none')
                body = (torch.arange(canvas_size, device=device) < meta['body_tokens']) & c_valid
                tail = (~body) & c_valid
                meta['body_ce_sum'] = float(ce[body].sum())
                meta['body_count'] = int(body.sum())
                meta['tail_ce_sum'] = float(ce[tail].sum())
                meta['tail_count'] = int(tail.sum())
        return torch.stack(row_losses).mean()

    mean_loss = _run_view_loss(current_noisy, current_corrupted, sample_times, record_diagnostics=diagnostics)
    if complementary_mask:
        comp_noisy: list[torch.Tensor] = []
        comp_corrupted: list[torch.Tensor] = []
        comp_times: list[float] = []
        for b in range(batch_size):
            c_mask = ~current_corrupted[b]
            if mask_id is not None:
                fill = torch.full_like(current_targets[b], int(mask_id))
            else:
                fill = torch.randint(0, vocab_size, current_targets[b].shape, device=device)
            comp_noisy.append(torch.where(c_mask, fill, current_targets[b]))
            comp_corrupted.append(c_mask)
            comp_times.append(max(time_eps, 1.0 - sample_times[b]))
        comp_loss = _run_view_loss(comp_noisy, comp_corrupted, comp_times, record_diagnostics=False)
        mean_loss = 0.5 * (mean_loss + comp_loss)

    parts = {"decoder_loss": float(mean_loss.detach())}
    if diagnostics:
        parts['rows'] = diagnostic_rows
    return mean_loss, parts


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
    token_shift: bool = False,
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

    position_ids = (valid.long().cumsum(dim=-1) - 1).clamp(min=0)
    keep = canvas.shape[1] if (not token_shift and _supports_logits_to_keep(model)) else None
    out = _diffusion_forward(
        model, tokens, signal, history_len,
        _gemma_masks(allowed, model_dtype, valid=valid, sliding_window=window, model=model),
        position_ids,
        logits_to_keep=keep,
    )
    if keep is not None:
        return out.logits
    return _extract_canvas_logits(
        out.logits, [history_len], history_len, canvas.shape[1], token_shift=token_shift
    )


def denoise_logits(
    model,
    prompt: torch.Tensor,
    canvas: torch.Tensor,
    *,
    pad_id: int,
    token_shift: bool = False,
) -> torch.Tensor:
    """Two-pass convenience wrapper when no prior ``sc_embeddings`` are carried."""
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    canvas_len = canvas.numel()
    zero_sc = torch.zeros(1, canvas_len, hidden_size, device=prompt.device, dtype=_token_embedding(model).weight.dtype)
    with torch.no_grad():
        first_logits = denoise_step_logits(
            model, prompt, canvas, sc_embeddings=zero_sc, pad_id=pad_id, token_shift=token_shift
        )
    adapter = get_diffusion_adapter(model)
    signal = _embed_distribution(_token_embedding(model), first_logits, scale=adapter.embed_scale(model))
    return denoise_step_logits(
        model, prompt, canvas, sc_embeddings=signal, pad_id=pad_id, token_shift=token_shift
    )
