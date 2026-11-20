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
    model_kwargs: dict | None = None,
):
    """Delegate model-specific embedding/PLE/hook behavior to the resolved ``DiffusionModelAdapter``."""
    conditioner = _get_conditioner(model)
    adapter = get_diffusion_adapter(model)
    extra = {"logits_to_keep": int(logits_to_keep)} if (logits_to_keep is not None and _supports_logits_to_keep(model)) else {}
    if model_kwargs:
        extra.update(model_kwargs)
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
    multimodal_inputs: list[dict | None] | None = None,
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
    selected_offsets: list[int] = []
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
        selected_offsets.append(prefix_len)
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

    adapter = get_diffusion_adapter(model)
    current_valid_batch = torch.stack(current_valids, dim=0)
    target_batch = torch.stack(current_targets, dim=0)
    valid_batch = torch.cat([history_valid_batch, current_valid_batch], dim=1)
    position_ids = (valid_batch.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = adapter.build_position_ids(
        model,
        prompts,
        multimodal_inputs,
        position_ids,
        clean_lengths=history_lengths,
        canvas_start=max_hist,
        noisy_start_offsets=selected_offsets,
        noisy_lengths=[canvas_size] * batch_size,
        pad_id=pad_id,
    )
    mm_batch = adapter.prepare_multimodal_batch(model, multimodal_inputs, device)

    total = max_hist + canvas_size
    allowed = torch.zeros((batch_size, 1, total, total), dtype=torch.bool, device=device)
    hist_causal = _allow_mask(history_valid_batch, current_valid_batch[:, :0], canvas_bidirectional=False)
    allowed[:, :, :max_hist, :max_hist] = hist_causal
    full_canvas_allow = _allow_mask(history_valid_batch, current_valid_batch, canvas_bidirectional=True)
    allowed[:, :, max_hist:, :] = full_canvas_allow[:, :, max_hist:, :]
    attn_masks = _gemma_masks(allowed, model_dtype, valid=valid_batch, sliding_window=window, model=model)

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
                sub_mm_kwargs = mm_batch.kwargs_for_rows(None, allow_grad=False)
            else:
                sub_tokens = view_tokens.index_select(0, active_idx)
                sub_zero = zero_signal.index_select(0, active_idx)
                sub_masks = (
                    {k: v.index_select(0, active_idx) for k, v in attn_masks.items()}
                    if isinstance(attn_masks, dict)
                    else attn_masks.index_select(0, active_idx)
                )
                sub_pos = adapter.select_batch_positions(position_ids, active_idx)
                sub_hist_lens = [history_lengths[int(i)] for i in active_idx.tolist()]
                sub_mm_kwargs = mm_batch.kwargs_for_rows(active_idx.tolist(), allow_grad=False)
            with torch.no_grad():
                first = _diffusion_forward(
                    ref,
                    sub_tokens,
                    sub_zero,
                    max_hist,
                    sub_masks,
                    sub_pos,
                    logits_to_keep=ref_keep,
                    model_kwargs=sub_mm_kwargs,
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
        dec_mm_kwargs = mm_batch.kwargs_for_rows(None, allow_grad=True)
        decoder = _diffusion_forward(
            model,
            view_tokens,
            signal,
            max_hist,
            attn_masks,
            position_ids,
            logits_to_keep=dec_keep,
            model_kwargs=dec_mm_kwargs,
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


def _build_all_blocks_mask(
    clean_valid: torch.Tensor,
    noisy_valid: torch.Tensor,
    prompt_lengths: list[int],
    start_blocks: list[int],
    num_sel_blocks: list[int],
    canvas_size: int,
) -> torch.Tensor:
    """Build exact staircase block-diffusion attention mask ``[B, 1, C_len + N_len, C_len + N_len]``.

    - Clean prefix tokens attend causally within valid clean positions and NEVER see noisy blocks.
    - Noisy block ``m`` (corresponding to response block ``blk = start_blocks[b] + m``) attends
      bidirectionally within its own ``canvas_size`` tokens and sees ONLY clean tokens in
      ``[0, prompt_lengths[b] + blk * canvas_size)``.
    """
    batch, max_clean = clean_valid.shape
    max_noisy = noisy_valid.shape[1]
    device = clean_valid.device
    total = max_clean + max_noisy
    allowed = torch.zeros((batch, 1, total, total), dtype=torch.bool, device=device)

    # Clean-to-clean: standard causal over valid clean keys.
    if max_clean > 0:
        cq = torch.arange(max_clean, device=device)[:, None]
        ck = torch.arange(max_clean, device=device)[None, :]
        allowed[:, 0, :max_clean, :max_clean] = (cq >= ck)[None, :, :] & clean_valid[:, None, :]

    # Noisy-to-clean and noisy-to-noisy per row.
    clean_pos = torch.arange(max_clean, device=device)
    for b in range(batch):
        p_len = int(prompt_lengths[b])
        s_blk = int(start_blocks[b])
        n_blk = int(num_sel_blocks[b])
        c_val = clean_valid[b]
        for m in range(n_blk):
            q_start = max_clean + m * canvas_size
            q_end = q_start + canvas_size
            q_valid = noisy_valid[b, m * canvas_size : (m + 1) * canvas_size]
            # Noisy block (s_blk + m) sees prompt + clean blocks 0 .. (s_blk + m - 1).
            prefix_limit = p_len + (s_blk + m) * canvas_size
            if max_clean > 0 and prefix_limit > 0:
                k_clean_ok = c_val & (clean_pos < prefix_limit)
                allowed[b, 0, q_start:q_end, :max_clean] = q_valid[:, None] & k_clean_ok[None, :]
            # Strictly bidirectional within the same block; zero attention to other noisy blocks.
            allowed[b, 0, q_start:q_end, q_start:q_end] = q_valid[:, None] & q_valid[None, :]

    return allowed


def _topk_forward_kl(
    student_logits: torch.Tensor,
    teacher_top_p: torch.Tensor,
    teacher_top_idx: torch.Tensor,
    *,
    kl_temp: float = 1.0,
) -> torch.Tensor:
    """Rao-Blackwellized sparse top-K forward KL ``D_KL(p_AR || p_diff)`` per token (>= 0)."""
    temp = max(float(kl_temp), 1e-4)
    student_log_probs = F.log_softmax(student_logits.float() / temp, dim=-1)
    gathered = student_log_probs.gather(-1, teacher_top_idx)
    kl = (teacher_top_p * (torch.log(teacher_top_p.clamp_min(1e-12)) - gathered)).sum(dim=-1)
    return kl * (temp * temp)


@torch.no_grad()
def _compute_ar_teacher_topk(
    ar_teacher,
    clean_full_seqs: list[torch.Tensor],
    prompt_lengths: list[int],
    start_blocks: list[int],
    num_sel_blocks: list[int],
    canvas_size: int,
    *,
    pad_id: int,
    kl_topk: int,
    kl_temp: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run one causal AR teacher forward pass and extract top-K soft targets for each row's selected blocks."""
    device = clean_full_seqs[0].device
    batch = len(clean_full_seqs)
    max_len = max(seq.numel() for seq in clean_full_seqs)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    for b, seq in enumerate(clean_full_seqs):
        L = seq.numel()
        input_ids[b, :L] = seq
        attn_mask[b, :L] = (seq != pad_id).long()

    min_start = min(int(p_len) + int(s_blk) * canvas_size - 1 for p_len, s_blk in zip(prompt_lengths, start_blocks))
    keep_n = max_len - min_start
    use_keep = _supports_logits_to_keep(ar_teacher) and 0 < keep_n < max_len
    extra = {"logits_to_keep": keep_n} if use_keep else {}
    out = ar_teacher(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, **extra)
    logits = out.logits
    offset = (max_len - keep_n) if use_keep else 0
    vocab_size = logits.shape[-1]
    k = min(max(int(kl_topk), 1), vocab_size)
    temp = max(float(kl_temp), 1e-4)

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(batch):
        p_len = int(prompt_lengths[b])
        s_blk = int(start_blocks[b])
        n_blk = int(num_sel_blocks[b])
        n_tok = n_blk * canvas_size
        # Causal shift: position (p_len + j - 1) predicts response token j.
        start_pos = p_len + s_blk * canvas_size - 1 - offset
        end_pos = start_pos + n_tok
        row_logits = logits[b, start_pos:end_pos].float()
        top_vals, top_idx = torch.topk(row_logits, k=k, dim=-1)
        top_p = torch.softmax(top_vals / temp, dim=-1)
        results.append((top_p, top_idx))
    return results


def diffusion_all_blocks_step(
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
    loss_mode: str | DiffusionObjective = "canvas-mean",
    corruption_mode: str | CorruptionProcess = "uniform",
    mask_id: int | None = None,
    max_train_blocks: int = 6,
    ar_teacher=None,
    kl_weight: float = 0.0,
    kl_topk: int = 64,
    kl_temp: float = 1.0,
    residual_mode: str = "standard",
    ref_model=None,
    multimodal_inputs: list[dict | None] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Multi-block parallel section-4 conversion step with optional frozen AR-teacher top-K KL distillation.

    Trains up to ``max_train_blocks`` contiguous canvases per row in ONE staircase forward pass,
    mathematically identical to running each block separately with full clean history, while
    optionally adding Rao-Blackwellized forward KL distillation from a frozen causal AR teacher
    strictly on non-padded ``body`` tokens.
    """
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
    adapter = get_diffusion_adapter(model)
    token_embedding = _token_embedding(model)
    model_dtype = token_embedding.weight.dtype

    batch_size = len(prompts)
    first_indices = torch.arange(num_canvases, device=device) * canvas_size

    clean_prefix_seqs: list[torch.Tensor] = []
    clean_prefix_valids: list[torch.Tensor] = []
    clean_full_seqs: list[torch.Tensor] = []
    noisy_seqs: list[torch.Tensor] = []
    noisy_targets: list[torch.Tensor] = []
    noisy_valids: list[torch.Tensor] = []
    noisy_corrupted: list[torch.Tensor] = []
    noisy_pos_lists: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    start_blocks: list[int] = []
    num_sel_blocks: list[int] = []
    block_times_per_row: list[list[float]] = []
    block_body_lens_per_row: list[list[int]] = []
    diagnostic_rows: list[dict] = []

    use_kl = ar_teacher is not None and float(kl_weight) > 0.0

    for p_raw, r_raw in zip(prompts, responses):
        p = p_raw.to(device=device, dtype=torch.long).view(-1)
        r = r_raw.to(device=device, dtype=torch.long).view(-1)
        if p.numel() == 0:
            raise ValueError("Diffusion prompt is empty.")
        canvas, _, canvas_mask = _chunk(r, num_canvases, canvas_size, eos_id, pad_id)
        num_valid_blocks = int(canvas_mask[first_indices].sum().clamp(min=1).item())
        cap = max(1, int(max_train_blocks))
        if num_valid_blocks > cap:
            start_blk = int(torch.randint(0, num_valid_blocks - cap + 1, (), device=device).item())
            n_sel = cap
        else:
            start_blk = 0
            n_sel = num_valid_blocks

        # Last selected block index is (start_blk + n_sel - 1); clean prefix only needs blocks < last_sel!
        prefix_blocks = start_blk + n_sel - 1
        prefix_len = prefix_blocks * canvas_size
        p_len = int(p.numel())

        c_pref = torch.cat([p, canvas[:prefix_len]], dim=0) if prefix_len > 0 else p
        c_pref_valid = torch.cat([p != pad_id, canvas_mask[:prefix_len]], dim=0) if prefix_len > 0 else (p != pad_id)
        clean_prefix_seqs.append(c_pref)
        clean_prefix_valids.append(c_pref_valid)

        if use_kl:
            end_clean_len = (start_blk + n_sel) * canvas_size
            clean_full_seqs.append(torch.cat([p, canvas[:end_clean_len]], dim=0))

        # Corrupt each selected block independently with its own noise level t_k.
        row_noisy, row_targets, row_valids, row_corrupted, row_times, row_bodies = [], [], [], [], [], []
        for m in range(n_sel):
            blk_idx = start_blk + m
            sl = slice(blk_idx * canvas_size, (blk_idx + 1) * canvas_size)
            clean_blk = canvas[sl]
            valid_blk = canvas_mask[sl]
            cs = corruption.corrupt(
                clean_blk,
                vocab_size=vocab_size,
                time_eps=time_eps,
                time_min=time_min,
                time_max=time_max,
            )
            row_noisy.append(cs.noisy)
            row_targets.append(clean_blk)
            row_valids.append(valid_blk)
            row_corrupted.append(cs.corrupted)
            row_times.append(cs.time)
            row_bodies.append(max(0, min(canvas_size, r.numel() - blk_idx * canvas_size)))

        noisy_seqs.append(torch.cat(row_noisy, dim=0))
        noisy_targets.append(torch.cat(row_targets, dim=0))
        noisy_valids.append(torch.cat(row_valids, dim=0))
        noisy_corrupted.append(torch.cat(row_corrupted, dim=0))
        # Exact RoPE positions for noisy blocks: start at (valid_prompt_tokens + start_blk * canvas_size).
        valid_p_count = int((p != pad_id).sum().item())
        noisy_pos_lists.append(
            torch.arange(n_sel * canvas_size, device=device, dtype=torch.long)
            + valid_p_count
            + start_blk * canvas_size
        )
        prompt_lengths.append(p_len)
        start_blocks.append(start_blk)
        num_sel_blocks.append(n_sel)
        block_times_per_row.append(row_times)
        block_body_lens_per_row.append(row_bodies)

    ar_targets = (
        _compute_ar_teacher_topk(
            ar_teacher,
            clean_full_seqs,
            prompt_lengths,
            start_blocks,
            num_sel_blocks,
            canvas_size,
            pad_id=pad_id,
            kl_topk=kl_topk,
            kl_temp=kl_temp,
        )
        if use_kl
        else None
    )

    max_clean = max(seq.numel() for seq in clean_prefix_seqs)
    max_noisy = max(seq.numel() for seq in noisy_seqs)
    total = max_clean + max_noisy

    tokens = torch.full((batch_size, total), pad_id, dtype=torch.long, device=device)
    clean_valid_batch = torch.zeros((batch_size, max_clean), dtype=torch.bool, device=device)
    noisy_valid_batch = torch.zeros((batch_size, max_noisy), dtype=torch.bool, device=device)
    position_ids = torch.zeros((batch_size, total), dtype=torch.long, device=device)

    for b in range(batch_size):
        clen = clean_prefix_seqs[b].numel()
        nlen = noisy_seqs[b].numel()
        tokens[b, :clen] = clean_prefix_seqs[b]
        clean_valid_batch[b, :clen] = clean_prefix_valids[b]
        if clen > 0:
            position_ids[b, :clen] = (clean_prefix_valids[b].long().cumsum(dim=-1) - 1).clamp(min=0)
        tokens[b, max_clean : max_clean + nlen] = noisy_seqs[b]
        noisy_valid_batch[b, :nlen] = noisy_valids[b]
        position_ids[b, max_clean : max_clean + nlen] = noisy_pos_lists[b]

    position_ids = adapter.build_position_ids(
        model,
        prompts,
        multimodal_inputs,
        position_ids,
        clean_lengths=[seq.numel() for seq in clean_prefix_seqs],
        canvas_start=max_clean,
        noisy_start_offsets=[s_blk * canvas_size for s_blk in start_blocks],
        noisy_lengths=[seq.numel() for seq in noisy_seqs],
        pad_id=pad_id,
    )
    mm_batch = adapter.prepare_multimodal_batch(model, multimodal_inputs, device)

    allowed = _build_all_blocks_mask(
        clean_valid_batch,
        noisy_valid_batch,
        prompt_lengths,
        start_blocks,
        num_sel_blocks,
        canvas_size,
    )
    valid_batch = torch.cat([clean_valid_batch, noisy_valid_batch], dim=1)
    attn_masks = _gemma_masks(allowed, model_dtype, valid=valid_batch, sliding_window=window, model=model)

    zero_signal = torch.zeros((batch_size, max_noisy, hidden_size), device=device, dtype=model_dtype)
    ref = ref_model if (residual_mode == "rcd" and ref_model is not None) else model
    ref_keep = max_noisy if _supports_logits_to_keep(ref) else None
    use_signal = torch.rand((batch_size, 1, 1), device=device) < 0.5
    signal = zero_signal
    if bool(use_signal.any()):
        active_idx = use_signal.view(-1).nonzero(as_tuple=False).flatten()
        if active_idx.numel() == batch_size:
            sub_tokens, sub_zero, sub_masks, sub_pos = tokens, zero_signal, attn_masks, position_ids
            sub_mm_kwargs = mm_batch.kwargs_for_rows(None, allow_grad=False)
        else:
            sub_tokens = tokens.index_select(0, active_idx)
            sub_zero = zero_signal.index_select(0, active_idx)
            sub_masks = (
                {k: v.index_select(0, active_idx) for k, v in attn_masks.items()}
                if isinstance(attn_masks, dict)
                else attn_masks.index_select(0, active_idx)
            )
            sub_pos = adapter.select_batch_positions(position_ids, active_idx)
            sub_mm_kwargs = mm_batch.kwargs_for_rows(active_idx.tolist(), allow_grad=False)
        with torch.no_grad():
            first = _diffusion_forward(
                ref,
                sub_tokens,
                sub_zero,
                max_clean,
                sub_masks,
                sub_pos,
                logits_to_keep=ref_keep,
                model_kwargs=sub_mm_kwargs,
            )
        first_logits = first.logits if ref_keep is not None else first.logits[:, max_clean:]
        if residual_mode == "rcd":
            embedded_first = rcd_residual_signal(
                token_embedding,
                first_logits,
                sub_tokens[:, max_clean:],
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

    dec_keep = max_noisy if _supports_logits_to_keep(model) else None
    dec_mm_kwargs = mm_batch.kwargs_for_rows(None, allow_grad=True)
    decoder = _diffusion_forward(
        model,
        tokens,
        signal,
        max_clean,
        attn_masks,
        position_ids,
        logits_to_keep=dec_keep,
        model_kwargs=dec_mm_kwargs,
    )
    noisy_logits = decoder.logits if dec_keep is not None else decoder.logits[:, max_clean:]

    block_ce_losses: list[torch.Tensor] = []
    block_kl_losses: list[torch.Tensor] = []
    total_blocks = 0

    for b in range(batch_size):
        n_sel = num_sel_blocks[b]
        s_blk = start_blocks[b]
        t_row = noisy_targets[b]
        v_row = noisy_valids[b]
        c_row = noisy_corrupted[b]
        ar_row = ar_targets[b] if ar_targets is not None else None

        row_body_ce = 0.0
        row_body_cnt = 0
        row_tail_ce = 0.0
        row_tail_cnt = 0
        row_changed_sum = 0.0

        for m in range(n_sel):
            sl = slice(m * canvas_size, (m + 1) * canvas_size)
            b_logits = noisy_logits[b, sl]
            b_targets = t_row[sl]
            b_valid = v_row[sl]
            b_corr = c_row[sl]
            b_body_len = block_body_lens_per_row[b][m]
            b_time = block_times_per_row[b][m]
            if not bool(b_valid.any()):
                raise RuntimeError("Diffusion decoder selected no canvas tokens.")
            if not torch.isfinite(b_logits[b_valid]).all():
                raise RuntimeError("Decoder logits are non-finite on the selected canvas.")

            ce_loss = objective.compute(
                b_logits,
                b_targets,
                b_valid,
                b_corr,
                body_tokens=b_body_len,
                time=b_time,
            )
            block_ce_losses.append(ce_loss)
            total_blocks += 1

            if ar_row is not None and b_body_len > 0:
                body_mask = (torch.arange(canvas_size, device=device) < b_body_len) & b_valid
                if bool(body_mask.any()):
                    top_p, top_idx = ar_row
                    kl_tok = _topk_forward_kl(
                        b_logits[body_mask],
                        top_p[sl][body_mask],
                        top_idx[sl][body_mask],
                        kl_temp=kl_temp,
                    )
                    block_kl_losses.append(kl_tok.mean())

            if diagnostics:
                ce_vec = F.cross_entropy(b_logits.detach().float(), b_targets, reduction="none")
                body_mask = (torch.arange(canvas_size, device=device) < b_body_len) & b_valid
                tail_mask = (~body_mask) & b_valid
                row_body_ce += float(ce_vec[body_mask].sum())
                row_body_cnt += int(body_mask.sum())
                row_tail_ce += float(ce_vec[tail_mask].sum())
                row_tail_cnt += int(tail_mask.sum())
                row_changed_sum += float(b_corr.float().mean())

        if diagnostics:
            diagnostic_rows.append(
                dict(
                    time=sum(block_times_per_row[b]) / max(n_sel, 1),
                    selected_block=s_blk,
                    num_blocks=n_sel,
                    body_tokens=row_body_cnt,
                    changed_fraction=row_changed_sum / max(n_sel, 1),
                    body_ce_sum=row_body_ce,
                    body_count=row_body_cnt,
                    tail_ce_sum=row_tail_ce,
                    tail_count=row_tail_cnt,
                )
            )

    mean_ce = torch.stack(block_ce_losses).mean()
    if block_kl_losses and float(kl_weight) > 0.0:
        mean_kl = torch.stack(block_kl_losses).mean()
        total_loss = mean_ce + float(kl_weight) * mean_kl
        kl_val = float(mean_kl.detach())
    else:
        total_loss = mean_ce
        kl_val = 0.0

    parts = {
        "decoder_loss": float(total_loss.detach()),
        "ce_loss": float(mean_ce.detach()),
        "kl_loss": kl_val,
        "trained_blocks": total_blocks,
        "trained_tokens": total_blocks * canvas_size,
    }
    if diagnostics:
        parts["rows"] = diagnostic_rows
    return total_loss, parts


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
    multimodal_input: dict | None = None,
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
    token_embedding = _token_embedding(model)
    model_dtype = token_embedding.weight.dtype
    signal = sc_embeddings.to(device=tokens.device, dtype=model_dtype)

    position_ids = (valid.long().cumsum(dim=-1) - 1).clamp(min=0)
    adapter = get_diffusion_adapter(model)
    mm_kwargs: dict = {}
    if multimodal_input is not None:
        position_ids = adapter.build_position_ids(
            model,
            [prompt[0]],
            [multimodal_input],
            position_ids,
            clean_lengths=[history_len],
            canvas_start=history_len,
            noisy_start_offsets=[0],
            noisy_lengths=[canvas.shape[1]],
            pad_id=pad_id,
        )
        mm_batch = adapter.prepare_multimodal_batch(model, [multimodal_input], tokens.device)
        mm_kwargs = mm_batch.kwargs_for_rows(None, allow_grad=False)
    keep = canvas.shape[1] if (not token_shift and _supports_logits_to_keep(model)) else None
    out = _diffusion_forward(
        model,
        tokens,
        signal,
        history_len,
        _gemma_masks(allowed, model_dtype, valid=valid, sliding_window=window, model=model),
        position_ids,
        logits_to_keep=keep,
        model_kwargs=mm_kwargs,
    )
    if keep is not None:
        return out.logits
    return _extract_canvas_logits(
        out.logits, [history_len], history_len, canvas.shape[1], token_shift=token_shift
    )


def supports_history_kv_cache(model) -> bool:
    """Delegate KV-cache eligibility check to the model's ``DiffusionModelAdapter``."""
    return get_diffusion_adapter(model).supports_history_kv_cache(model)


class DiffusionHistoryCache:
    """Model-agnostic causal KV cache over clean prefix/history for O(1)-history denoising steps."""

    def __init__(self, model, *, pad_id: int):
        from transformers import DynamicCache

        self.model = model
        self.adapter = get_diffusion_adapter(model)
        self.pad_id = int(pad_id)
        self.past_key_values = DynamicCache()
        self.history_valid: torch.Tensor | None = None
        self._pos_state: dict = {}

    @property
    def seq_length(self) -> int:
        return int(self.past_key_values.get_seq_length())

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, *, multimodal_input: dict | None = None) -> None:
        """Append clean causal tokens (optionally multimodal) into the KV cache."""
        tokens_2d = tokens.view(1, -1)
        seq_len = int(tokens_2d.shape[1])
        if seq_len == 0:
            return
        device = tokens_2d.device
        valid_2d = tokens_2d != self.pad_id
        past_len = self.seq_length
        hist_valid = valid_2d if self.history_valid is None else torch.cat([self.history_valid, valid_2d], dim=1)

        allowed = torch.zeros((1, 1, seq_len, past_len + seq_len), dtype=torch.bool, device=device)
        if past_len > 0:
            allowed[:, :, :, :past_len] = self.history_valid[:, None, None, :]
        q = torch.arange(seq_len, device=device)[:, None]
        k = torch.arange(seq_len, device=device)[None, :]
        allowed[:, 0, :, past_len:] = (q >= k)[None, :, :] & valid_2d[:, None, :]

        model_dtype = _token_embedding(self.model).weight.dtype
        attn_mask = self.adapter.build_attention_mask(allowed, model_dtype)
        pos, mm_kwargs = self.adapter.build_cache_prefill_inputs(
            self.model, tokens_2d, valid_2d, multimodal_input, self._pos_state
        )

        extra = {"logits_to_keep": 1} if _supports_logits_to_keep(self.model) else {}
        extra.update(mm_kwargs)
        self.model(
            input_ids=tokens_2d,
            attention_mask=attn_mask,
            position_ids=pos,
            past_key_values=self.past_key_values,
            use_cache=True,
            **extra,
        )
        self.history_valid = hist_valid

    def denoise_step_logits(self, canvas: torch.Tensor, *, sc_embeddings: torch.Tensor) -> torch.Tensor:
        """Denoise one bidirectional canvas against the cached history without mutating the cache."""
        if self.history_valid is None:
            raise RuntimeError("DiffusionHistoryCache must be prefilled before denoise_step_logits.")
        canvas_2d = canvas.view(1, -1)
        canvas_len = int(canvas_2d.shape[1])
        device = canvas_2d.device
        past_len = self.seq_length

        allowed = torch.ones((1, 1, canvas_len, past_len + canvas_len), dtype=torch.bool, device=device)
        allowed[:, :, :, :past_len] = self.history_valid[:, None, None, :]

        model_dtype = _token_embedding(self.model).weight.dtype
        conditioner = _get_conditioner(self.model)
        attn_mask = self.adapter.build_attention_mask(allowed, model_dtype)
        pos = self.adapter.build_cache_canvas_positions(canvas_len, device, self._pos_state)

        signal = sc_embeddings.to(device=device, dtype=model_dtype)
        try:
            with self.adapter.forward_context(self.model, canvas_2d, signal, 0, conditioner):
                out = self.model(
                    input_ids=canvas_2d,
                    attention_mask=attn_mask,
                    position_ids=pos,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
        finally:
            to_remove = self.seq_length - past_len
            if to_remove > 0:
                self.past_key_values.crop(-to_remove)
        return out.logits


def denoise_logits(
    model,
    prompt: torch.Tensor,
    canvas: torch.Tensor,
    *,
    pad_id: int,
    token_shift: bool = False,
    multimodal_input: dict | None = None,
) -> torch.Tensor:
    """Two-pass convenience wrapper when no prior ``sc_embeddings`` are carried."""
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    canvas_len = canvas.numel()
    zero_sc = torch.zeros(1, canvas_len, hidden_size, device=prompt.device, dtype=_token_embedding(model).weight.dtype)
    if multimodal_input is not None and supports_history_kv_cache(model):
        cache = DiffusionHistoryCache(model, pad_id=pad_id)
        cache.prefill(prompt, multimodal_input=multimodal_input)
        with torch.no_grad():
            first_logits = cache.denoise_step_logits(canvas, sc_embeddings=zero_sc)
        adapter = get_diffusion_adapter(model)
        signal = _embed_distribution(_token_embedding(model), first_logits, scale=adapter.embed_scale(model))
        return cache.denoise_step_logits(canvas, sc_embeddings=signal)
    with torch.no_grad():
        first_logits = denoise_step_logits(
            model,
            prompt,
            canvas,
            sc_embeddings=zero_sc,
            pad_id=pad_id,
            token_shift=token_shift,
            multimodal_input=multimodal_input,
        )
    adapter = get_diffusion_adapter(model)
    signal = _embed_distribution(_token_embedding(model), first_logits, scale=adapter.embed_scale(model))
    return denoise_step_logits(
        model,
        prompt,
        canvas,
        sc_embeddings=signal,
        pad_id=pad_id,
        token_shift=token_shift,
        multimodal_input=multimodal_input,
    )
