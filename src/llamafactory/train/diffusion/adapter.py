"""Model-agnostic adapter interface for section-4 AR-to-diffusion conversion."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import MethodType
from typing import Any, Protocol

import torch
from torch import nn


def _token_embedding(model) -> nn.Module:
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


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


class DiffusionModelAdapter(Protocol):
    """Contract isolating backbone-specific embedding, masking, and hook quirks."""

    name: str

    def embed_scale(self, model) -> float:
        """Multiplier applied to token embeddings (sqrt(d) for Gemma, 1.0 for standard LMs)."""
        ...

    def prepare_training(self, model) -> None:
        """Unlock model-specific buffers/parameters before optimizer creation."""
        ...

    def build_attention_mask(
        self,
        allowed: torch.Tensor,
        dtype: torch.dtype,
        *,
        valid: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> Any:
        """Convert boolean [B, 1, Q, K] allow-mask into the model's expected attention_mask."""
        ...

    @contextmanager
    def forward_context(self, model, tokens: torch.Tensor, signal: torch.Tensor, canvas_start: int, conditioner: nn.Module):
        """Inject self-conditioning and any architecture-specific hooks for one forward pass."""
        ...

    def resolve_special_ids(self, tokenizer) -> tuple[int, int, tuple[int, ...]]:
        """Return (pad_id, eos_id, stop_tokens) without pad/eos collision."""
        ...


class StandardCausalAdapter:
    """Adapter for standard GQA/Dense causal LMs (Qwen3, Llama, Mistral, etc.)."""

    name = "standard_causal"

    def embed_scale(self, model) -> float:
        del model
        return 1.0

    def prepare_training(self, model) -> None:
        del model

    def build_attention_mask(
        self,
        allowed: torch.Tensor,
        dtype: torch.dtype,
        *,
        valid: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del valid, sliding_window
        # Passing a dict bypasses transformers' create_causal_mask override in Qwen3/Llama.
        return {"full_attention": _additive_mask(allowed, dtype)}

    @contextmanager
    def forward_context(self, model, tokens: torch.Tensor, signal: torch.Tensor, canvas_start: int, conditioner: nn.Module):
        del tokens
        embedding = _token_embedding(model)
        padding_idx = embedding.padding_idx

        def embedding_hook(_module, _inputs, embeddings):
            canvas = embeddings[:, canvas_start:]
            conditioned = conditioner(canvas, signal.to(device=canvas.device, dtype=canvas.dtype))
            return torch.cat([embeddings[:, :canvas_start], conditioned], dim=1)

        handle = embedding.register_forward_hook(embedding_hook)
        try:
            embedding.padding_idx = None
            yield
        finally:
            handle.remove()
            embedding.padding_idx = padding_idx

    def resolve_special_ids(self, tokenizer) -> tuple[int, int, tuple[int, ...]]:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        im_end = vocab.get("<|im_end|>")
        endoftext = vocab.get("<|endoftext|>")
        pad_id = endoftext if isinstance(endoftext, int) else (tokenizer.pad_token_id or 0)
        eos_id = im_end if isinstance(im_end, int) else (tokenizer.eos_token_id or 1)
        if eos_id == pad_id:
            eos_id = 1 if pad_id != 1 else 2
        stop_tokens = tuple(
            sorted({int(x) for x in (eos_id, tokenizer.eos_token_id) if x is not None and int(x) != pad_id})
        )
        return int(pad_id), int(eos_id), stop_tokens or (int(eos_id),)


class Gemma4Adapter:
    """Adapter for Gemma 4 (E2B/E4B/26B-A4B) preserving official PLE, skip_scale, and sliding-band rules."""

    name = "gemma4"

    def embed_scale(self, model) -> float:
        text_config = getattr(model.config, "text_config", model.config)
        hidden = int(getattr(text_config, "hidden_size", 0) or _token_embedding(model).weight.shape[-1])
        return float(hidden) ** 0.5

    def prepare_training(self, model) -> None:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(base, "model", None)
        text = getattr(inner, "language_model", inner)
        for layer in text.layers:
            if hasattr(layer, "layer_scalar") and not isinstance(layer.layer_scalar, nn.Parameter):
                layer.layer_scalar = nn.Parameter(layer.layer_scalar)

    def build_attention_mask(
        self,
        allowed: torch.Tensor,
        dtype: torch.dtype,
        *,
        valid: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> dict[str, torch.Tensor]:
        full = _additive_mask(allowed, dtype)
        if sliding_window is None or valid is None:
            return {"full_attention": full, "sliding_attention": full}
        positions = (valid.long().cumsum(dim=-1) - 1).clamp(min=0)
        distance = (positions[:, :, None] - positions[:, None, :]).abs()
        band = distance < sliding_window
        while band.ndim < allowed.ndim:
            band = band[:, None]
        return {"full_attention": full, "sliding_attention": _additive_mask(allowed & band, dtype)}

    @contextmanager
    def forward_context(self, model, tokens: torch.Tensor, signal: torch.Tensor, canvas_start: int, conditioner: nn.Module):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(base, "model", None)
        text = getattr(inner, "language_model", inner)
        embedding = _token_embedding(model)
        padding_idx = embedding.padding_idx
        raw_embeddings = None
        project = getattr(text, "project_per_layer_inputs", None)
        had_instance_method = "project_per_layer_inputs" in vars(text)
        saved_instance_method = vars(text).get("project_per_layer_inputs")
        placeholder = getattr(inner, "get_placeholder_mask", None)
        had_placeholder = "get_placeholder_mask" in vars(inner)
        saved_placeholder = vars(inner).get("get_placeholder_mask")

        def text_only_placeholders(_self, input_ids=None, inputs_embeds=None, **kwargs):
            ids = input_ids if input_ids is not None else tokens
            empty = torch.zeros_like(ids, dtype=torch.bool)
            return empty, empty, empty

        def embedding_hook(_module, _inputs, embeddings):
            nonlocal raw_embeddings
            raw_embeddings = embeddings
            canvas = embeddings[:, canvas_start:]
            conditioned = conditioner(canvas, signal.to(device=canvas.device, dtype=canvas.dtype))
            return torch.cat([embeddings[:, :canvas_start], conditioned], dim=1)

        def project_raw(_self, inputs_embeds, per_layer_inputs=None):
            if raw_embeddings is None:
                raise RuntimeError("Diffusion PLE projection ran before token embedding.")
            canvas_ple = project(raw_embeddings[:, canvas_start:], None)
            if canvas_start == 0:
                return canvas_ple
            history_identity = None if per_layer_inputs is None else per_layer_inputs[:, :canvas_start]
            history_ple = project(raw_embeddings[:, :canvas_start], history_identity)
            return torch.cat([history_ple, canvas_ple], dim=1)

        with ExitStack():
            handle = embedding.register_forward_hook(embedding_hook)
            try:
                embedding.padding_idx = None
                if project is not None:
                    text.project_per_layer_inputs = MethodType(project_raw, text)
                if placeholder is not None:
                    inner.get_placeholder_mask = MethodType(text_only_placeholders, inner)
                yield
            finally:
                handle.remove()
                embedding.padding_idx = padding_idx
                if placeholder is not None:
                    if had_placeholder:
                        inner.get_placeholder_mask = saved_placeholder
                    else:
                        delattr(inner, "get_placeholder_mask")
                if project is not None:
                    if had_instance_method:
                        text.project_per_layer_inputs = saved_instance_method
                    else:
                        delattr(text, "project_per_layer_inputs")

    def resolve_special_ids(self, tokenizer) -> tuple[int, int, tuple[int, ...]]:
        pad_id = int(tokenizer.pad_token_id or 0)
        eos_id = int(tokenizer.eos_token_id or 1)
        return pad_id, eos_id, (eos_id, 106)


_ADAPTER_REGISTRY: dict[str, DiffusionModelAdapter] = {
    "gemma4": Gemma4Adapter(),
    "gemma4_text": Gemma4Adapter(),
    "gemma3": Gemma4Adapter(),
    "gemma3_text": Gemma4Adapter(),
    "qwen3": StandardCausalAdapter(),
    "qwen2": StandardCausalAdapter(),
    "llama": StandardCausalAdapter(),
    "mistral": StandardCausalAdapter(),
}


def register_diffusion_adapter(model_type: str, adapter: DiffusionModelAdapter) -> None:
    """Register a custom ``DiffusionModelAdapter`` for a Hugging Face ``model_type``."""
    _ADAPTER_REGISTRY[model_type.lower()] = adapter


def get_diffusion_adapter(model) -> DiffusionModelAdapter:
    """Resolve the ``DiffusionModelAdapter`` for ``model`` without hardcoding logic in loss.py."""
    text_config = getattr(model.config, "text_config", model.config)
    model_type = str(getattr(text_config, "model_type", "") or getattr(model.config, "model_type", "")).lower()
    if model_type in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[model_type]
    return StandardCausalAdapter()
