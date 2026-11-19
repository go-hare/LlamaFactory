"""Pluggable corruption processes and training objectives for AR-to-diffusion conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CorruptedSample:
    noisy: torch.Tensor
    corrupted: torch.Tensor
    time: float


class CorruptionProcess(Protocol):
    """Forward noising process transforming clean canvas tokens into noisy inputs."""

    name: str

    def corrupt(
        self,
        clean: torch.Tensor,
        *,
        vocab_size: int,
        time_eps: float = 1e-4,
        time_min: float | None = None,
        time_max: float | None = None,
    ) -> CorruptedSample:
        ...


class DiffusionObjective(Protocol):
    """Loss objective evaluated on the denoiser's predicted logits for one canvas."""

    name: str

    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        body_tokens: int,
        time: float,
    ) -> torch.Tensor:
        ...


def _sample_clipped_time(
    device: torch.device,
    *,
    time_eps: float,
    time_min: float | None,
    time_max: float | None,
) -> float:
    low = time_eps if time_min is None else max(float(time_min), float(time_eps))
    high = (1.0 - time_eps) if time_max is None else min(float(time_max), 1.0 - float(time_eps))
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid clipped time interval [{low}, {high}]")
    return float(torch.empty((), device=device).uniform_(low, high).item())


class UniformNoiseCorruption:
    """Official DiffusionGemma forward process: replace tokens uniformly from vocabulary with prob t."""

    name = "uniform"

    def corrupt(
        self,
        clean: torch.Tensor,
        *,
        vocab_size: int,
        time_eps: float = 1e-4,
        time_min: float | None = None,
        time_max: float | None = None,
    ) -> CorruptedSample:
        device = clean.device
        t = _sample_clipped_time(device, time_eps=time_eps, time_min=time_min, time_max=time_max)
        noise = torch.randint(0, vocab_size, clean.shape, device=device)
        move = torch.rand(clean.shape, device=device) < t
        return CorruptedSample(noisy=torch.where(move, noise, clean), corrupted=move, time=t)


class MaskTokenCorruption:
    """Absorbing-state masking process (BD3-LMs/MDLM style): replace tokens with ``mask_id`` with prob t."""

    name = "mask"

    def __init__(self, mask_id: int):
        self.mask_id = int(mask_id)

    def corrupt(
        self,
        clean: torch.Tensor,
        *,
        vocab_size: int,
        time_eps: float = 1e-4,
        time_min: float | None = None,
        time_max: float | None = None,
    ) -> CorruptedSample:
        del vocab_size
        device = clean.device
        t = _sample_clipped_time(device, time_eps=time_eps, time_min=time_min, time_max=time_max)
        move = torch.rand(clean.shape, device=device) < t
        masked = torch.full_like(clean, self.mask_id)
        return CorruptedSample(noisy=torch.where(move, masked, clean), corrupted=move, time=t)


class CanvasMeanObjective:
    """Official section-4 objective: unweighted cross-entropy over all valid canvas positions."""

    name = "canvas-mean"

    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        body_tokens: int,
        time: float,
    ) -> torch.Tensor:
        del corrupted, body_tokens, time
        return F.cross_entropy(logits[valid], targets[valid])


class CorruptedOnlyObjective:
    """Compute cross-entropy only on corrupted/masked positions (falls back to valid if none corrupted)."""

    name = "corrupted-only"

    def __init__(self, *, time_weight: bool = False):
        self.time_weight = bool(time_weight)

    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        body_tokens: int,
        time: float,
    ) -> torch.Tensor:
        del body_tokens
        mask = valid & corrupted
        if not bool(mask.any()):
            mask = valid
        loss = F.cross_entropy(logits[mask], targets[mask])
        if self.time_weight:
            loss = loss / max(float(time), 1e-3)
        return loss


class BodyTailBalancedObjective:
    """Diagnostic objective weighting response body and EOS-padding tail equally (0.5 / 0.5)."""

    name = "body-tail-balanced"

    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        body_tokens: int,
        time: float,
    ) -> torch.Tensor:
        del corrupted, time
        body = (torch.arange(len(targets), device=targets.device) < body_tokens) & valid
        tail = (~body) & valid
        terms = [F.cross_entropy(logits[mask], targets[mask]) for mask in (body, tail) if bool(mask.any())]
        return torch.stack(terms).mean()


def resolve_corruption(spec: str | CorruptionProcess | None, *, mask_id: int | None = None) -> CorruptionProcess:
    if spec is None or spec == "uniform":
        return UniformNoiseCorruption()
    if spec == "mask":
        if mask_id is None:
            raise ValueError("MaskTokenCorruption requires an explicit mask_id.")
        return MaskTokenCorruption(mask_id)
    if isinstance(spec, str):
        raise ValueError(f"Unknown corruption mode: {spec}")
    return spec


def resolve_objective(spec: str | DiffusionObjective | None) -> DiffusionObjective:
    if spec is None or spec == "canvas-mean":
        return CanvasMeanObjective()
    if spec == "body-tail-balanced":
        return BodyTailBalancedObjective()
    if spec == "corrupted-only":
        return CorruptedOnlyObjective(time_weight=False)
    if spec == "corrupted-only-weighted":
        return CorruptedOnlyObjective(time_weight=True)
    if isinstance(spec, str):
        raise ValueError(f"Unknown loss mode: {spec}")
    return spec
