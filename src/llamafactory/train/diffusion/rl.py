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

import torch
import torch.nn.functional as F

from .loss import _embed_distribution, _token_embedding, denoise_step_logits


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
    mask = torch.zeros_like(keep)
    mask.scatter_(1, order, keep)
    if is_final_step:
        mask = torch.ones_like(mask)
    drawn = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(canvas.shape)
    noise = torch.randint(0, vocab_size, canvas.shape, device=canvas.device)
    return torch.where(mask, drawn, noise), scaled, entropy


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise one canvas from uniform noise with carried self-conditioning and adaptive stopping."""
    text_config = getattr(model.config, "text_config", model.config)
    if vocab_size <= 0:
        vocab_size = int(getattr(text_config, "vocab_size", 0) or model.config.vocab_size)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    token_embedding = _token_embedding(model)
    canvas = torch.randint(0, vocab_size, (1, canvas_size), device=prompt.device)
    sc_embeddings = torch.zeros(1, canvas_size, hidden_size, device=prompt.device, dtype=token_embedding.weight.dtype)
    mean_entropy = torch.zeros((), device=prompt.device)
    for step in range(steps):
        is_final_step = (step + 1) >= steps
        logits = denoise_step_logits(model, prompt, canvas[0], sc_embeddings=sc_embeddings, pad_id=pad_id)
        next_canvas, shaped_logits, token_entropy = _commit(
            logits,
            canvas,
            temperature=_temperature(step, steps),
            entropy_bound=0.1,
            vocab_size=vocab_size,
            is_final_step=is_final_step,
        )
        mean_entropy = token_entropy.mean()
        sc_embeddings = _embed_distribution(token_embedding, shaped_logits)
        stable = bool((shaped_logits.argmax(dim=-1) == canvas).all())
        confident = bool(mean_entropy <= entropy_threshold)
        if early_stop and stable and confident:
            canvas = shaped_logits.argmax(dim=-1)
            break
        canvas = next_canvas
    return canvas, mean_entropy.detach()


def sample_response(
    model,
    prompt: torch.Tensor,
    *,
    max_canvases: int = 4,
    steps: int = 48,
    canvas_size: int = 256,
    pad_id: int = 0,
    stop_tokens: tuple[int, ...] = (1, 106),
    vocab_size: int = 0,
    early_stop: bool = True,
) -> torch.Tensor:
    """Block-autoregressive generation across canvases, truncating at the first stop token."""
    running_prompt = prompt.view(-1)
    chunks: list[torch.Tensor] = []
    stop_set = set(stop_tokens)
    for _ in range(max_canvases):
        canvas, _ = sample_canvas(
            model,
            running_prompt,
            steps=steps,
            canvas_size=canvas_size,
            pad_id=pad_id,
            vocab_size=vocab_size,
            early_stop=early_stop,
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
