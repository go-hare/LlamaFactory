"""Second stage of DiffusionGemma: sampler distillation with a reward.

The report does not publish the loss. What it does publish is the sampler and
the goal. The model is its own teacher: it denoises a canvas for several steps
at a high step budget and mild temperature, and that trajectory is the
reference. A student denoises the same prompt in fewer steps and is trained to
match the teacher's distribution. A verifiable reward (an answer string that
must appear) raises the probability of the teacher's tokens when the trajectory
is right and lowers it when it is wrong.

Temperature follows the published inference schedule, 0.8 down to 0.4.
Positions are committed by cumulative entropy, as in the official sampler, and
the rest are drawn again from the vocabulary.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from .adapter import get_diffusion_adapter
from .loss import (
    DiffusionHistoryCache,
    _embed_distribution,
    _get_conditioner,
    _token_embedding,
    _topk_forward_kl,
    denoise_step_logits,
    rcd_residual_signal,
    supports_history_kv_cache,
)
from .objective import resolve_objective


def _apply_local_repetition_penalty(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    *,
    penalty: float = 1.0,
    exempt_tokens: tuple[int, ...] = (),
) -> torch.Tensor:
    """Dampen local token-run lock-in (e.g. `,,,,,` / `、、、、`) when a non-stop token repeats >=2x in +/-2 neighbors."""
    if penalty <= 1.0 or canvas.shape[1] < 3:
        return logits
    bsz, length = canvas.shape
    padded = torch.full((bsz, length + 4), -1, dtype=canvas.dtype, device=canvas.device)
    padded[:, 2 : 2 + length] = canvas
    m2, m1, p1, p2 = padded[:, :-4], padded[:, 1:-3], padded[:, 3:-1], padded[:, 4:]
    pairs = [
        (m1, (m1 >= 0) & ((m1 == m2) | (m1 == p1) | (m1 == p2))),
        (p1, (p1 >= 0) & ((p1 == m2) | (p1 == p2)) & (p1 != m1)),
        (m2, (m2 >= 0) & (m2 == p2) & (m2 != m1) & (m2 != p1)),
    ]
    out = logits.clone()
    vocab = out.shape[-1]
    for tok_ids, mask in pairs:
        mask = mask & (tok_ids < vocab)
        for ex in exempt_tokens:
            mask = mask & (tok_ids != int(ex))
        if not bool(mask.any()):
            continue
        safe_ids = tok_ids.clamp(min=0, max=vocab - 1).unsqueeze(-1)
        cur = out.gather(-1, safe_ids).squeeze(-1)
        penalized = torch.where(cur > 0, cur / penalty, cur * penalty)
        updated = torch.where(mask, penalized, cur)
        out.scatter_(-1, safe_ids, updated.unsqueeze(-1))
    return out


def _temperature(step: int, total: int, start: float = 0.8, end: float = 0.4, exponent: float = 1.0) -> float:
    if total <= 0:
        return end
    noise_proportion = 1.0 - step / total
    fraction = 1.0 - (1.0 - noise_proportion) ** exponent
    return end + (start - end) * fraction


def _commit(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    *,
    temperature: float,
    entropy_bound: float,
    vocab_size: int,
    is_final_step: bool = False,
    defer_adjacent_duplicates: bool = False,
    exempt_tokens: tuple[int, ...] = (),
    min_keep: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the most confident tokens. Draw the rest again from the vocabulary unless on the final step."""
    scaled = logits.float() / max(temperature, 1e-6)
    log_probs = torch.log_softmax(scaled, dim=-1)
    probs = torch.exp(log_probs)
    safe_log_probs = torch.where(probs == 0, torch.zeros_like(log_probs), log_probs)
    entropy = -(safe_log_probs * probs).sum(dim=-1)
    order = entropy.argsort(dim=-1)
    sorted_entropy = entropy.gather(1, order)
    keep = (sorted_entropy.cumsum(dim=-1) - sorted_entropy) <= entropy_bound
    if min_keep > 0:
        keep[:, : min(int(min_keep), keep.shape[1])] = True
    mask = torch.zeros_like(keep)
    mask.scatter_(1, order, keep)
    if is_final_step:
        mask = torch.ones_like(mask)
    drawn = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(canvas.shape)
    if defer_adjacent_duplicates and not is_final_step and canvas.shape[1] >= 2:
        # Break same-step parallel stuttering (e.g. `realtoror`, `沙发上上`):
        # if pos i-1 newly changed to token v on this step AND pos i simultaneously sampled v,
        # keep i-1 and defer i by 1 step so i sees i-1 cleanly next step (preserving true reduplications).
        left_newly_changed = canvas[:, :-1] != drawn[:, :-1]
        same_as_left = drawn[:, 1:] == drawn[:, :-1]
        both_committing = mask[:, :-1] & mask[:, 1:]
        conflict = both_committing & left_newly_changed & same_as_left
        for ex in exempt_tokens:
            conflict = conflict & (drawn[:, 1:] != int(ex))
        mask[:, 1:] = mask[:, 1:] & (~conflict)
    noise = torch.randint(0, vocab_size, canvas.shape, device=canvas.device)
    return torch.where(mask, drawn, noise), scaled, entropy


def _ddim_step(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    *,
    time_now: float,
    time_next: float,
    temperature: float = 0.7,
    vocab_size: int,
    protect_stop_tokens: tuple[int, ...] = (),
    protect_stop_time: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official Hackable Diffusion ``DiscreteDDIMStep`` (STAY / NOISE / CLEAN posterior).

    Used by Google's SFT evaluation (``ar_eval.py``) prior to stage-2 sampler
    distillation, avoiding the 0.1 cumulative-entropy bottleneck on un-distilled checkpoints.
    """
    scaled = logits.float() / max(temperature, 1e-6)
    log_probs = torch.log_softmax(scaled, dim=-1)
    probs = torch.exp(log_probs)
    safe_log_probs = torch.where(probs == 0, torch.zeros_like(log_probs), log_probs)
    entropy = -(safe_log_probs * probs).sum(dim=-1)

    x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(canvas.shape)
    x_noise = torch.randint(0, vocab_size, canvas.shape, device=canvas.device)

    alpha_t = 1.0 - float(time_now)
    alpha_s = 1.0 - float(time_next)
    ratio = alpha_t / max(alpha_s, 1e-12)
    pi = 1.0 / float(vocab_size)
    ones = torch.ones_like(canvas, dtype=torch.float32)

    p_stay = ones * (ratio * (1.0 - alpha_s) * pi)
    p_noise = ones * ((1.0 - ratio) * (1.0 - alpha_s) * pi)
    p_clean = ones * ((1.0 - ratio) * alpha_s * pi)

    same = x0 == canvas
    if protect_stop_tokens and float(time_now) > float(protect_stop_time):
        stop_mask = torch.zeros_like(same)
        for sid in protect_stop_tokens:
            stop_mask = stop_mask | (canvas == sid)
        same = same & (~stop_mask)
    same_f = same.float()
    p_stay = p_stay + same_f * (ratio * alpha_s + p_clean)
    p_clean = (1.0 - same_f) * p_clean

    weights = torch.stack([p_stay, p_noise, p_clean], dim=-1).clamp(min=1e-20)
    route = torch.multinomial(weights.view(-1, 3), 1).view(canvas.shape)
    next_canvas = torch.where(route == 0, canvas, torch.where(route == 1, x_noise, x0))
    return next_canvas, scaled, entropy


def _mask_unmask_step(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    *,
    mask_id: int,
    remaining_steps: int,
    temperature: float = 0.7,
    protect_stop_tokens: tuple[int, ...] = (),
    protect_stop_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BD3-LMs / MDLM absorbing-state unmasking step with SUBS zero-mask-prob constraint.

    Already-unmasked tokens remain locked; among currently masked positions, the
    most confident ``ceil(num_masked / remaining_steps)`` tokens are unmasked.
    """
    scaled = logits.float() / max(temperature, 1e-6)
    # SUBS parameterization: never emit [MASK] as a clean token prediction.
    scaled[:, :, int(mask_id)] = -float("inf")
    log_probs = torch.log_softmax(scaled, dim=-1)
    probs = torch.exp(log_probs)
    safe_log_probs = torch.where(probs == 0, torch.zeros_like(log_probs), log_probs)
    entropy = -(safe_log_probs * probs).sum(dim=-1)

    drawn = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(canvas.shape)
    is_masked = canvas == int(mask_id)
    next_canvas = canvas.clone()

    for b in range(canvas.shape[0]):
        masked_pos = is_masked[b].nonzero(as_tuple=False).flatten()
        n_masked = int(masked_pos.numel())
        if n_masked == 0:
            continue
        k = n_masked if remaining_steps <= 1 else max(1, (n_masked + remaining_steps - 1) // remaining_steps)
        scores = entropy[b, masked_pos]
        if protect_stop_active and protect_stop_tokens:
            stop_hit = torch.zeros_like(scores, dtype=torch.bool)
            cand = drawn[b, masked_pos]
            for sid in protect_stop_tokens:
                stop_hit = stop_hit | (cand == int(sid))
            # Push premature stop-token candidates to the back while keeping k unlockable if non-stops exist.
            scores = torch.where(stop_hit, scores + 1e6, scores)
        chosen_local = scores.topk(min(k, n_masked), largest=False).indices
        chosen_pos = masked_pos[chosen_local]
        next_canvas[b, chosen_pos] = drawn[b, chosen_pos]

    return next_canvas, scaled, entropy


def _remdm_step(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    *,
    mask_id: int,
    time_now: float,
    time_next: float,
    temperature: float = 0.7,
    remask_eta: float = 0.08,
    protect_stop_tokens: tuple[int, ...] = (),
    protect_stop_active: bool = False,
    is_final_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ReMDM confidence-weighted remasking step for absorbing-state [MASK] diffusion."""
    scaled = logits.float() / max(temperature, 1e-6)
    scaled[:, :, int(mask_id)] = -float("inf")
    if protect_stop_active and protect_stop_tokens:
        for sid in protect_stop_tokens:
            scaled[:, :, int(sid)] = -float("inf")
    log_probs = torch.log_softmax(scaled, dim=-1)
    probs = torch.exp(log_probs)
    safe_log_probs = torch.where(probs == 0, torch.zeros_like(log_probs), log_probs)
    entropy = -(safe_log_probs * probs).sum(dim=-1)

    drawn = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(canvas.shape)
    is_masked = canvas == int(mask_id)
    if is_final_step or time_next <= 1e-6:
        next_canvas = torch.where(is_masked, drawn, canvas)
        return next_canvas, scaled, entropy

    alpha_t = max(1.0 - float(time_now), 1e-6)
    alpha_s = min(max(1.0 - float(time_next), 0.0), 1.0)
    max_sigma = (1.0 - alpha_s) / alpha_t
    sigma_cap = min(max(float(remask_eta), 0.0), max_sigma)

    # Remask unmasked tokens proportionally to model disagreement (1 - p(x_t)).
    safe_tokens = torch.where(is_masked, torch.zeros_like(canvas), canvas)
    current_prob = probs.gather(-1, safe_tokens.unsqueeze(-1)).squeeze(-1)
    disagreement = 1.0 - current_prob
    p_remask = (sigma_cap * disagreement).clamp(0.0, min(1.0, max_sigma))
    remask_draw = torch.rand_like(p_remask) < p_remask

    p_unmask = min(1.0, max(0.0, (alpha_s - (1.0 - sigma_cap) * alpha_t) / max(1.0 - alpha_t, 1e-6)))
    unmask_draw = torch.rand_like(p_remask) < p_unmask

    mask_tensor = torch.full_like(canvas, int(mask_id))
    next_masked = torch.where(unmask_draw, drawn, canvas)
    next_unmasked = torch.where(remask_draw, mask_tensor, canvas)
    next_canvas = torch.where(is_masked, next_masked, next_unmasked)
    return next_canvas, scaled, entropy


def sample_canvas(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 48,
    canvas_size: int = 256,
    pad_id: int = 0,
    vocab_size: int = 0,
    early_stop: bool = True,
    entropy_threshold: float = 0.005,
    sampler: str = "entropy",
    temperature: float = 0.7,
    protect_stop_tokens: tuple[int, ...] = (),
    protect_stop_time: float = 0.0,
    mask_id: int | None = None,
    remask_eta: float = 0.08,
    multimodal_input: dict | None = None,
    history_cache: DiffusionHistoryCache | None = None,
    use_kv_cache: bool = False,
    repetition_penalty: float = 1.0,
    exempt_tokens: tuple[int, ...] = (),
    defer_adjacent_duplicates: bool = False,
    progressive_commit: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise one canvas from uniform noise or [MASK] prior with carried self-conditioning."""
    if sampler not in ("entropy", "ddim", "mask", "remdm", "rcd"):
        raise ValueError(f"Unknown diffusion sampler: {sampler}")
    if sampler in ("mask", "remdm", "rcd") and mask_id is None:
        raise ValueError(f"sampler={sampler!r} requires an explicit mask_id.")
    text_config = getattr(model.config, "text_config", model.config)
    if vocab_size <= 0:
        vocab_size = int(getattr(text_config, "vocab_size", 0) or model.config.vocab_size)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    token_embedding = _token_embedding(model)
    embed_scale = get_diffusion_adapter(model).embed_scale(model)
    if sampler in ("mask", "remdm", "rcd"):
        canvas = torch.full((1, canvas_size), int(mask_id), dtype=torch.long, device=prompt.device)
    else:
        canvas = torch.randint(0, vocab_size, (1, canvas_size), device=prompt.device)
    sc_embeddings = torch.zeros(1, canvas_size, hidden_size, device=prompt.device, dtype=token_embedding.weight.dtype)
    mean_entropy = torch.zeros((), device=prompt.device)
    ddim_times = torch.linspace(1.0 - 1e-6, 0.0, steps + 1, device=prompt.device) if sampler in ("ddim", "remdm", "rcd") else None
    active_cache = history_cache
    if active_cache is None and use_kv_cache and supports_history_kv_cache(model):
        active_cache = DiffusionHistoryCache(model, pad_id=pad_id)
        active_cache.prefill(prompt, multimodal_input=multimodal_input)
    rep_exempt = tuple({int(pad_id), *(int(t) for t in protect_stop_tokens), *(int(t) for t in exempt_tokens)})
    temp_scale = max(float(temperature), 1e-4) / 0.7
    for step in range(steps):
        is_final_step = (step + 1) >= steps
        if active_cache is not None:
            logits = active_cache.denoise_step_logits(canvas[0], sc_embeddings=sc_embeddings)
        elif multimodal_input is not None:
            logits = denoise_step_logits(
                model,
                prompt,
                canvas[0],
                sc_embeddings=sc_embeddings,
                pad_id=pad_id,
                multimodal_input=multimodal_input,
            )
        else:
            logits = denoise_step_logits(
                model,
                prompt,
                canvas[0],
                sc_embeddings=sc_embeddings,
                pad_id=pad_id,
            )
        if repetition_penalty > 1.0:
            logits = _apply_local_repetition_penalty(
                logits, canvas, penalty=repetition_penalty, exempt_tokens=rep_exempt
            )
        if sampler == "entropy":
            commit_extra: dict = {}
            if defer_adjacent_duplicates:
                commit_extra["defer_adjacent_duplicates"] = True
                commit_extra["exempt_tokens"] = rep_exempt
            if progressive_commit:
                commit_extra["min_keep"] = math.ceil((step + 1) * canvas_size / max(steps, 1))
            next_canvas, shaped_logits, token_entropy = _commit(
                logits,
                canvas,
                temperature=_temperature(step, steps, start=0.8 * temp_scale, end=0.4 * temp_scale),
                entropy_bound=0.1,
                vocab_size=vocab_size,
                is_final_step=is_final_step,
                **commit_extra,
            )
        elif sampler == "mask":
            progress = 1.0 - (step / max(steps, 1))
            next_canvas, shaped_logits, token_entropy = _mask_unmask_step(
                logits,
                canvas,
                mask_id=int(mask_id),
                remaining_steps=steps - step,
                temperature=temperature,
                protect_stop_tokens=protect_stop_tokens,
                protect_stop_active=bool(protect_stop_tokens) and (progress > float(protect_stop_time)),
            )
        elif sampler in ("remdm", "rcd"):
            time_now = float(ddim_times[step])
            next_canvas, shaped_logits, token_entropy = _remdm_step(
                logits,
                canvas,
                mask_id=int(mask_id),
                time_now=time_now,
                time_next=float(ddim_times[step + 1]),
                temperature=temperature,
                remask_eta=remask_eta,
                protect_stop_tokens=protect_stop_tokens,
                protect_stop_active=bool(protect_stop_tokens) and (time_now > float(protect_stop_time)),
                is_final_step=is_final_step,
            )
        else:
            next_canvas, shaped_logits, token_entropy = _ddim_step(
                logits,
                canvas,
                time_now=float(ddim_times[step]),
                time_next=float(ddim_times[step + 1]),
                temperature=temperature,
                vocab_size=vocab_size,
                protect_stop_tokens=protect_stop_tokens,
                protect_stop_time=protect_stop_time,
            )
        mean_entropy = token_entropy.mean()
        if sampler == "rcd":
            sc_embeddings = rcd_residual_signal(
                token_embedding,
                shaped_logits,
                next_canvas,
                mask_id=int(mask_id),
                scale=embed_scale,
            )
        else:
            sc_embeddings = _embed_distribution(token_embedding, shaped_logits, scale=embed_scale)
        stable = bool((shaped_logits.argmax(dim=-1) == canvas).all())
        confident = bool(mean_entropy <= entropy_threshold)
        if early_stop and stable and confident and not (sampler in ("mask", "remdm", "rcd") and bool((next_canvas == int(mask_id)).any())):
            canvas = next_canvas
            break
        canvas = next_canvas
    return canvas, mean_entropy.detach()


def sample_response(
    model,
    prompt: torch.Tensor,
    *,
    max_canvases: int = 18,
    steps: int = 48,
    canvas_size: int = 256,
    pad_id: int = 0,
    stop_tokens: tuple[int, ...] = (1, 106),
    vocab_size: int = 0,
    early_stop: bool = True,
    sampler: str = "entropy",
    temperature: float = 0.7,
    protect_stop_time: float = 0.0,
    mask_id: int | None = None,
    remask_eta: float = 0.08,
    multimodal_input: dict | None = None,
    use_kv_cache: bool = False,
    repetition_penalty: float = 1.0,
    defer_adjacent_duplicates: bool = False,
    progressive_commit: bool = False,
) -> torch.Tensor:
    """Block-autoregressive generation across canvases, truncating at the first stop token."""
    running_prompt = prompt.view(-1)
    chunks: list[torch.Tensor] = []
    stop_set = set(stop_tokens)
    history_cache = None
    if use_kv_cache and supports_history_kv_cache(model):
        history_cache = DiffusionHistoryCache(model, pad_id=pad_id)
        history_cache.prefill(running_prompt, multimodal_input=multimodal_input)
    for _ in range(max_canvases):
        canvas, _ = sample_canvas(
            model,
            running_prompt,
            steps=steps,
            canvas_size=canvas_size,
            pad_id=pad_id,
            vocab_size=vocab_size,
            early_stop=early_stop,
            sampler=sampler,
            temperature=temperature,
            protect_stop_tokens=stop_tokens if protect_stop_time > 0 else (),
            protect_stop_time=protect_stop_time,
            mask_id=mask_id,
            remask_eta=remask_eta,
            multimodal_input=multimodal_input if history_cache is None else None,
            history_cache=history_cache,
            repetition_penalty=repetition_penalty,
            exempt_tokens=stop_tokens,
            defer_adjacent_duplicates=defer_adjacent_duplicates,
            progressive_commit=progressive_commit,
        )
        row = canvas[0]
        hit_idx = None
        for idx, tid in enumerate(row.tolist()):
            if tid in stop_set:
                hit_idx = idx
                break
        if hit_idx is not None:
            chunks.append(row[: hit_idx + 1])
            break
        chunks.append(row)
        if history_cache is not None:
            history_cache.prefill(row)
        else:
            running_prompt = torch.cat([running_prompt, row], dim=0)
    return torch.cat(chunks, dim=0) if chunks else running_prompt.new_zeros((0,))


def sd_rl_step(
    model,
    prompt: torch.Tensor,
    *,
    answer: str,
    tokenizer,
    canvas_size: int,
    pad_id: int,
    teacher_steps: int,
    student_steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One prompt. The teacher samples without a gradient. The student matches it.

    ``answer`` is the string a correct canvas must contain. A hit gives reward
    +1, a miss gives -1. Rows with an empty answer get no reward term.
    """
    text_config = getattr(model.config, "text_config", model.config)
    vocab_size = int(getattr(text_config, "vocab_size", 0) or model.config.vocab_size)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        teacher, entropy = sample_canvas(
            model, prompt, steps=teacher_steps, canvas_size=canvas_size, pad_id=pad_id, vocab_size=vocab_size
        )
    if was_training:
        model.train()

    text = tokenizer.decode(teacher[0].tolist(), skip_special_tokens=True)
    reward = 0.0 if not answer else (1.0 if answer in text else -1.0)

    # The student starts from the same noise level as the teacher's last steps
    # and is scored against the teacher's committed tokens.
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    token_embedding = _token_embedding(model)
    student = torch.randint(0, vocab_size, teacher.shape, device=prompt.device)
    sc_embeddings = torch.zeros(1, canvas_size, hidden_size, device=prompt.device, dtype=token_embedding.weight.dtype)
    total = torch.zeros((), device=prompt.device)
    for step in range(student_steps):
        logits = denoise_step_logits(model, prompt, student[0], sc_embeddings=sc_embeddings, pad_id=pad_id)
        total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher.reshape(-1))
        with torch.no_grad():
            is_final = (step + 1) >= student_steps
            student, shaped_logits, _ = _commit(
                logits.detach(),
                student,
                temperature=_temperature(step, student_steps),
                entropy_bound=0.1,
                vocab_size=vocab_size,
                is_final_step=is_final,
            )
            sc_embeddings = _embed_distribution(token_embedding, shaped_logits)
    distill = total / student_steps
    # Reward only moves the last student prediction. A wrong trajectory is
    # pushed down, a right one is pulled up.
    last = denoise_step_logits(model, prompt, student[0], sc_embeddings=sc_embeddings, pad_id=pad_id)
    policy = F.cross_entropy(last.reshape(-1, last.shape[-1]), teacher.reshape(-1))
    loss = distill + (-reward) * policy
    return loss, {"reward": reward, "distill_loss": float(distill.detach()), "entropy": float(entropy)}


@contextmanager
def _swap_conditioner(model, replacement):
    if replacement is None:
        yield
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    holder = None
    for cand in (model, base, getattr(base, "model", None)):
        if cand is not None and hasattr(cand, "self_conditioner") and getattr(cand, "self_conditioner") is not None:
            holder = cand
            break
    if holder is None:
        yield
        return
    saved = holder.self_conditioner
    holder.self_conditioner = replacement
    try:
        yield
    finally:
        holder.self_conditioner = saved


def few_step_distill_step(
    model,
    prompt: torch.Tensor,
    response: torch.Tensor,
    *,
    canvas_size: int = 128,
    num_canvases: int = 18,
    pad_id: int = 0,
    eos_id: int = 1,
    student_steps: int = 16,
    teacher_substeps: int = 4,
    train_points: int = 3,
    loss_mode: str = "tail-capped",
    kl_weight: float = 0.5,
    kl_topk: int = 64,
    kl_temp: float = 1.0,
    temperature: float = 0.7,
    teacher_conditioner=None,
    multimodal_input: dict | None = None,
    backward_per_point: bool = False,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-2 few-step sampler distillation: guided entropy trajectory + N-to-1 progressive teacher refinement.

    1. Rolls out the student's exact ``student_steps`` progressive-commit entropy trajectory on a
       ground-truth block (using ``DiffusionHistoryCache`` for O(1) vision/history cost), capturing
       structured intermediate states ``(x_k, sc_k)`` without phrasing drift.
    2. From ``train_points`` stratified points along the trajectory, runs ``teacher_substeps`` careful
       steps of the frozen Stage-1 teacher (via PEFT ``disable_adapter()`` + ``teacher_conditioner``)
       starting from the student's exact state ``(x_k, sc_k)``, giving a soft top-K refinement target
       strictly on uncommitted positions.
    3. Trains the student on ``tail-capped`` CE + sparse forward KL so 1 student step matches
       ``teacher_substeps`` Stage-1 steps.
    """
    device = _token_embedding(model).weight.device
    text_config = getattr(model.config, "text_config", model.config)
    vocab_size = int(getattr(text_config, "vocab_size", 0) or model.config.vocab_size)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    token_embedding = _token_embedding(model)
    model_dtype = token_embedding.weight.dtype
    adapter = get_diffusion_adapter(model)
    embed_scale = adapter.embed_scale(model)
    objective = resolve_objective(loss_mode)

    p = prompt.to(device=device, dtype=torch.long).view(-1)
    r = response.to(device=device, dtype=torch.long).view(-1)
    if p.numel() == 0:
        raise ValueError("Diffusion prompt is empty.")

    n_valid_blocks = max(1, min(num_canvases, (int(r.numel()) + canvas_size - 1) // canvas_size))
    blk = int(torch.randint(0, n_valid_blocks, (), device=device).item())
    begin = blk * canvas_size
    body = r[begin : begin + canvas_size]
    body_tokens = int(body.numel())
    clean = r.new_full((canvas_size,), eos_id)
    if body_tokens > 0:
        clean[:body_tokens] = body
    history = torch.cat([p, r[:begin]], dim=0) if begin > 0 else p

    was_training = model.training
    use_cache = supports_history_kv_cache(model)
    temp_scale = max(float(temperature), 1e-4) / 0.7

    # 1. Guided progressive-commit entropy trajectory under torch.no_grad()
    traj: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    model.eval()
    with torch.no_grad():
        s_cache = None
        if use_cache:
            s_cache = DiffusionHistoryCache(model, pad_id=pad_id)
            s_cache.prefill(history, multimodal_input=multimodal_input)
        x = torch.randint(0, vocab_size, (1, canvas_size), device=device)
        sc = torch.zeros(1, canvas_size, hidden_size, device=device, dtype=model_dtype)
        committed = torch.zeros(1, canvas_size, dtype=torch.bool, device=device)
        for k in range(student_steps):
            traj.append((x.clone(), sc.clone(), committed.clone(), k))
            if s_cache is not None:
                logits_k = s_cache.denoise_step_logits(x[0], sc_embeddings=sc)
            else:
                logits_k = denoise_step_logits(
                    model, history, x[0], sc_embeddings=sc, pad_id=pad_id, multimodal_input=multimodal_input
                )
            step_temp = _temperature(k, student_steps, start=0.8 * temp_scale, end=0.4 * temp_scale)
            scaled = logits_k.float() / max(step_temp, 1e-6)
            log_p = torch.log_softmax(scaled, dim=-1)
            probs = torch.exp(log_p)
            safe_lp = torch.where(probs == 0, torch.zeros_like(log_p), log_p)
            ent = -(safe_lp * probs).sum(dim=-1)
            # Already-committed tokens stay committed; prioritize lowest-entropy uncommitted tokens.
            rank_ent = torch.where(committed, torch.full_like(ent, -1.0), ent)
            order = rank_ent.argsort(dim=-1)
            sorted_ent = ent.gather(1, order)
            keep = (sorted_ent.cumsum(dim=-1) - sorted_ent) <= 0.1
            min_keep = math.ceil((k + 1) * canvas_size / max(student_steps, 1))
            keep[:, : min(min_keep, canvas_size)] = True
            new_mask = torch.zeros_like(keep)
            new_mask.scatter_(1, order, keep)
            committed = committed | new_mask
            noise = torch.randint(0, vocab_size, (1, canvas_size), device=device)
            x = torch.where(committed, clean[None, :], noise)
            sc = _embed_distribution(token_embedding, scaled, scale=embed_scale)

    # 2. Stratified point selection across [0, student_steps)
    m_pts = max(1, min(int(train_points), student_steps))
    selected_indices: list[int] = []
    for s_idx in range(m_pts):
        lo = (s_idx * student_steps) // m_pts
        hi = max(lo + 1, ((s_idx + 1) * student_steps) // m_pts)
        selected_indices.append(int(torch.randint(lo, hi, (), device=device).item()))

    # 3. Frozen Stage-1 teacher N-to-1 progressive refinement from exact student states
    use_teacher_kl = float(kl_weight) > 0.0 and int(teacher_substeps) > 0 and (
        hasattr(model, "disable_adapter") or teacher_conditioner is not None
    )
    teacher_targets: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * m_pts
    if use_teacher_kl:
        disable_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        k_top = min(max(int(kl_topk), 1), vocab_size)
        t_temp = max(float(kl_temp), 1e-4)
        with torch.no_grad(), disable_ctx, _swap_conditioner(model, teacher_conditioner):
            t_cache = None
            if use_cache:
                t_cache = DiffusionHistoryCache(model, pad_id=pad_id)
                t_cache.prefill(history, multimodal_input=multimodal_input)
            for p_i, t_idx in enumerate(selected_indices):
                x_k, sc_k, _, k = traj[t_idx]
                tx = x_k.clone()
                tsc = sc_k.clone()
                last_t_logits = None
                for j in range(teacher_substeps):
                    if t_cache is not None:
                        last_t_logits = t_cache.denoise_step_logits(tx[0], sc_embeddings=tsc)
                    else:
                        last_t_logits = denoise_step_logits(
                            model, history, tx[0], sc_embeddings=tsc, pad_id=pad_id, multimodal_input=multimodal_input
                        )
                    frac = (k + (j + 1) / teacher_substeps) / max(student_steps, 1)
                    sub_keep = min(canvas_size, math.ceil(frac * canvas_size))
                    sub_temp = _temperature(
                        int(round(frac * student_steps)), student_steps, start=0.8 * temp_scale, end=0.4 * temp_scale
                    )
                    tx, t_scaled, _ = _commit(
                        last_t_logits,
                        tx,
                        temperature=sub_temp,
                        entropy_bound=0.1,
                        vocab_size=vocab_size,
                        is_final_step=False,
                        defer_adjacent_duplicates=True,
                        exempt_tokens=(pad_id, eos_id),
                        min_keep=sub_keep,
                    )
                    tsc = _embed_distribution(token_embedding, t_scaled, scale=embed_scale)
                top_vals, top_idx = torch.topk(last_t_logits[0].float(), k=k_top, dim=-1)
                top_p = torch.softmax(top_vals / t_temp, dim=-1)
                teacher_targets[p_i] = (top_p, top_idx)

    # 4. Student forward + loss on sampled trajectory points
    if was_training:
        model.train()
    valid = torch.ones(canvas_size, dtype=torch.bool, device=device)
    point_losses: list[torch.Tensor] = []
    ce_sum = 0.0
    kl_sum = 0.0
    for p_i, t_idx in enumerate(selected_indices):
        x_k, sc_k, committed_k, k = traj[t_idx]
        s_logits = denoise_step_logits(
            model,
            history,
            x_k[0],
            sc_embeddings=sc_k,
            pad_id=pad_id,
            multimodal_input=multimodal_input,
        )[0]
        uncommitted = ~committed_k[0]
        if not bool(uncommitted.any()):
            uncommitted = valid
        ce_loss = objective.compute(
            s_logits,
            clean,
            valid,
            uncommitted,
            body_tokens=body_tokens,
            time=max(1e-4, 1.0 - k / max(student_steps, 1)),
        )
        kl_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
        if teacher_targets[p_i] is not None:
            top_p, top_idx = teacher_targets[p_i]
            kl_loss = _topk_forward_kl(
                s_logits[uncommitted], top_p[uncommitted], top_idx[uncommitted], kl_temp=kl_temp
            ).mean()
        pt_loss = ce_loss + float(kl_weight) * kl_loss
        ce_sum += float(ce_loss.detach())
        kl_sum += float(kl_loss.detach())
        if backward_per_point:
            (pt_loss * (float(loss_scale) / m_pts)).backward()
            point_losses.append(pt_loss.detach())
        else:
            point_losses.append(pt_loss)

    mean_loss = torch.stack(point_losses).mean()
    return mean_loss, {
        "loss": float(mean_loss.detach()),
        "ce_loss": ce_sum / m_pts,
        "kl_loss": kl_sum / m_pts,
        "selected_block": blk,
        "body_tokens": body_tokens,
        "train_points": m_pts,
    }
