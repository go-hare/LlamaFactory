"""Model-agnostic adapter interface for section-4 AR-to-diffusion conversion."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import inspect
from types import MethodType, SimpleNamespace
from typing import Any, Protocol

import torch
from torch import nn


class MultimodalBatchHandle(Protocol):
    """Opaque per-batch multimodal context managed by a ``DiffusionModelAdapter``."""

    has_multimodal: bool

    def kwargs_for_rows(self, row_indices: list[int] | None = None, *, allow_grad: bool = True) -> dict[str, Any]:
        """Return model-specific forward kwargs for ``row_indices`` (or the full batch if ``None``)."""
        ...


class _EmptyMultimodalBatch:
    has_multimodal: bool = False

    def kwargs_for_rows(self, row_indices: list[int] | None = None, *, allow_grad: bool = True) -> dict[str, Any]:
        del row_indices, allow_grad
        return {}


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

    def supports_history_kv_cache(self, model) -> bool:
        """Whether this backbone supports croppable full-attention KV caching over clean history."""
        ...

    def freeze_vision(self, model) -> int:
        """Freeze any vision encoder parameters and return frozen param count (0 for text-only)."""
        ...

    def lora_target_modules(self, model, *, freeze_vision: bool = True) -> str | list[str]:
        """Return PEFT target_modules (isolating language layers from frozen vision encoders)."""
        ...

    def prepare_multimodal_batch(
        self, model, multimodal_inputs: list[dict | None] | None, device: torch.device
    ) -> MultimodalBatchHandle:
        """Precompute or package per-batch multimodal features for Pass-1/Pass-2 reuse."""
        ...

    def build_position_ids(
        self,
        model,
        prompts: list[torch.Tensor],
        multimodal_inputs: list[dict | None] | None,
        base_position_ids: torch.Tensor,
        *,
        clean_lengths: list[int],
        canvas_start: int,
        noisy_start_offsets: list[int],
        noisy_lengths: list[int],
        pad_id: int,
    ) -> torch.Tensor:
        """Build model-expected position_ids (1D [B, L] for causal LMs, or 3D/4D mRoPE for VLMs)."""
        ...

    def select_batch_positions(self, position_ids: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
        """Slice active batch rows from ``position_ids``."""
        ...

    def build_cache_prefill_inputs(
        self,
        model,
        tokens_2d: torch.Tensor,
        valid_2d: torch.Tensor,
        multimodal_input: dict | None,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build (position_ids, extra_model_kwargs) for one causal KV-cache prefill chunk."""
        ...

    def build_cache_canvas_positions(
        self, canvas_len: int, device: torch.device, state: dict[str, Any]
    ) -> torch.Tensor:
        """Build position_ids for a cached bidirectional canvas denoising step."""
        ...

    def load_backbone(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        processor_name: str | None = None,
    ) -> tuple[nn.Module, Any, Any]:
        """Load ``(model, tokenizer, processor_or_none)`` for this backbone family."""
        ...


class StandardCausalAdapter:
    """Adapter for standard GQA/Dense causal LMs (Qwen3, Qwen2, Llama, Mistral, MiniCPM5, etc.)."""

    name = "standard_causal"

    def __init__(self, *, dict_mask: bool = True):
        self.dict_mask = bool(dict_mask)

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
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        del valid, sliding_window
        mask = _additive_mask(allowed, dtype)
        # Qwen3/Gemma hybrid models take a layer-type dict; Llama/MiniCPM5 take a raw 4D tensor
        # which transformers' _preprocess_mask_arguments passes straight through unchanged.
        return {"full_attention": mask} if self.dict_mask else mask

    @contextmanager
    def forward_context(self, model, tokens: torch.Tensor, signal: torch.Tensor, canvas_start: int, conditioner: nn.Module):
        del tokens
        embedding = _token_embedding(model)
        padding_idx = embedding.padding_idx

        def embedding_hook(_module, _inputs, embeddings):
            canvas = embeddings[:, canvas_start:]
            conditioned = conditioner(canvas, signal.to(device=canvas.device, dtype=canvas.dtype))
            if canvas_start == 0:
                return conditioned
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
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        eos_list = [int(x) for x in raw_eos] if isinstance(raw_eos, (list, tuple)) else ([int(raw_eos)] if raw_eos is not None else [])
        raw_pad = getattr(tokenizer, "pad_token_id", None)
        pad_id = endoftext if isinstance(endoftext, int) else (int(raw_pad) if raw_pad is not None else 0)
        eos_id = im_end if isinstance(im_end, int) else (eos_list[0] if eos_list else 1)
        if eos_id == pad_id:
            eos_id = 1 if pad_id != 1 else 2
        stop_tokens = tuple(sorted({int(x) for x in [eos_id, *eos_list] if int(x) != pad_id}))
        return int(pad_id), int(eos_id), stop_tokens or (int(eos_id),)

    def supports_history_kv_cache(self, model) -> bool:
        text_config = getattr(model.config, "text_config", model.config)
        if getattr(text_config, "sliding_window", None) is not None:
            return False
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types and any(lt != "full_attention" for lt in layer_types):
            return False
        return True

    def freeze_vision(self, model) -> int:
        del model
        return 0

    def lora_target_modules(self, model, *, freeze_vision: bool = True) -> str | list[str]:
        del model, freeze_vision
        return "all-linear"

    def prepare_multimodal_batch(
        self, model, multimodal_inputs: list[dict | None] | None, device: torch.device
    ) -> MultimodalBatchHandle:
        del model, multimodal_inputs, device
        return _EmptyMultimodalBatch()

    def build_position_ids(
        self,
        model,
        prompts: list[torch.Tensor],
        multimodal_inputs: list[dict | None] | None,
        base_position_ids: torch.Tensor,
        *,
        clean_lengths: list[int],
        canvas_start: int,
        noisy_start_offsets: list[int],
        noisy_lengths: list[int],
        pad_id: int,
    ) -> torch.Tensor:
        del model, prompts, multimodal_inputs, clean_lengths, canvas_start, noisy_start_offsets, noisy_lengths, pad_id
        return base_position_ids

    def select_batch_positions(self, position_ids: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
        return position_ids.index_select(1 if position_ids.ndim == 3 else 0, active_idx)

    def build_cache_prefill_inputs(
        self,
        model,
        tokens_2d: torch.Tensor,
        valid_2d: torch.Tensor,
        multimodal_input: dict | None,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del model, tokens_2d, multimodal_input
        next_pos = int(state.get("next_text_pos", 0))
        pos = (valid_2d.long().cumsum(dim=-1) - 1).clamp(min=0) + next_pos
        state["next_text_pos"] = next_pos + int(valid_2d.sum().item())
        return pos, {}

    def build_cache_canvas_positions(
        self, canvas_len: int, device: torch.device, state: dict[str, Any]
    ) -> torch.Tensor:
        next_pos = int(state.get("next_text_pos", 0))
        return torch.arange(canvas_len, device=device, dtype=torch.long)[None, :] + next_pos

    def load_backbone(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        processor_name: str | None = None,
    ) -> tuple[nn.Module, Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(processor_name or model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        return model, tokenizer, None


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

    def supports_history_kv_cache(self, model) -> bool:
        del model
        return False

    def freeze_vision(self, model) -> int:
        return freeze_vision_encoder(model)

    def lora_target_modules(self, model, *, freeze_vision: bool = True) -> str | list[str]:
        del model, freeze_vision
        return "all-linear"

    def prepare_multimodal_batch(
        self, model, multimodal_inputs: list[dict | None] | None, device: torch.device
    ) -> MultimodalBatchHandle:
        del model, multimodal_inputs, device
        return _EmptyMultimodalBatch()

    def build_position_ids(
        self,
        model,
        prompts: list[torch.Tensor],
        multimodal_inputs: list[dict | None] | None,
        base_position_ids: torch.Tensor,
        *,
        clean_lengths: list[int],
        canvas_start: int,
        noisy_start_offsets: list[int],
        noisy_lengths: list[int],
        pad_id: int,
    ) -> torch.Tensor:
        del model, prompts, multimodal_inputs, clean_lengths, canvas_start, noisy_start_offsets, noisy_lengths, pad_id
        return base_position_ids

    def select_batch_positions(self, position_ids: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
        return position_ids.index_select(0, active_idx)

    def build_cache_prefill_inputs(
        self,
        model,
        tokens_2d: torch.Tensor,
        valid_2d: torch.Tensor,
        multimodal_input: dict | None,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError("Gemma4Adapter uses sliding-window attention and does not support DiffusionHistoryCache.")

    def build_cache_canvas_positions(
        self, canvas_len: int, device: torch.device, state: dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError("Gemma4Adapter uses sliding-window attention and does not support DiffusionHistoryCache.")

    def load_backbone(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        processor_name: str | None = None,
    ) -> tuple[nn.Module, Any, Any]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(processor_name or model_name)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(device)
        return model, tokenizer, processor


def _derive_qwen3_vl_token_types(tokens_1d: torch.Tensor, base_config, mm_entry: dict | None) -> torch.Tensor:
    n = int(tokens_1d.numel())
    if mm_entry is not None and mm_entry.get("mm_token_type_ids") is not None:
        provided = mm_entry["mm_token_type_ids"].to(device=tokens_1d.device, dtype=torch.int32).view(-1)
        if provided.numel() >= n:
            return provided[:n]
        padded = torch.zeros(n, dtype=torch.int32, device=tokens_1d.device)
        padded[: provided.numel()] = provided
        return padded
    mm_type = torch.zeros_like(tokens_1d, dtype=torch.int32)
    image_token_id = getattr(base_config, "image_token_id", None)
    video_token_id = getattr(base_config, "video_token_id", None)
    spatial_merge = int(getattr(getattr(base_config, "vision_config", None), "spatial_merge_size", 2) or 2)
    if image_token_id is not None and mm_entry is not None and mm_entry.get("image_grid_thw") is not None:
        grid = mm_entry["image_grid_thw"]
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        expected = int((grid.prod(-1) // (spatial_merge**2)).sum().item())
        hits = (tokens_1d == int(image_token_id)).nonzero(as_tuple=False).flatten()[:expected]
        if hits.numel() > 0:
            mm_type[hits] = 1
    if video_token_id is not None and mm_entry is not None and mm_entry.get("video_grid_thw") is not None:
        grid = mm_entry["video_grid_thw"]
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        expected = int((grid.prod(-1) // (spatial_merge**2)).sum().item())
        hits = (tokens_1d == int(video_token_id)).nonzero(as_tuple=False).flatten()[:expected]
        if hits.numel() > 0:
            mm_type[hits] = 2
    return mm_type


class _Qwen3VLMultimodalBatch:
    """Precomputes frozen Qwen3-VL vision + DeepStack features once per batch and slices per row subset."""

    def __init__(self, model, multimodal_inputs: list[dict | None] | None, device: torch.device):
        self.has_multimodal = False
        self.row_img_slices: list[tuple[int, int]] = []
        self.per_row_pv: list[torch.Tensor | None] = []
        self.per_row_grid: list[torch.Tensor | None] = []
        self.precomputed_image_out = None
        self.vit_trainable = False

        if not multimodal_inputs:
            return

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        visual = getattr(getattr(base, "model", None), "visual", None) or getattr(base, "visual", None)
        if visual is not None:
            self.vit_trainable = any(p.requires_grad for p in visual.parameters())

        pv_parts: list[torch.Tensor] = []
        grid_parts: list[torch.Tensor] = []
        cursor = 0
        for mm in multimodal_inputs:
            if mm is not None and mm.get("pixel_values") is not None and mm.get("image_grid_thw") is not None:
                pv = mm["pixel_values"].to(device=device)
                grid = mm["image_grid_thw"].to(device=device, dtype=torch.long)
                if grid.ndim == 1:
                    grid = grid.unsqueeze(0)
                n_imgs = int(grid.shape[0])
                self.row_img_slices.append((cursor, cursor + n_imgs))
                cursor += n_imgs
                self.per_row_pv.append(pv)
                self.per_row_grid.append(grid)
                pv_parts.append(pv)
                grid_parts.append(grid)
            else:
                self.row_img_slices.append((cursor, cursor))
                self.per_row_pv.append(None)
                self.per_row_grid.append(None)

        if pv_parts:
            self.has_multimodal = True
            try:
                supports_precomputed = "mm_encoder_outputs" in inspect.signature(base.forward).parameters
            except (TypeError, ValueError):
                supports_precomputed = False
            if supports_precomputed and hasattr(base, "get_image_features") and not self.vit_trainable:
                cat_pv = torch.cat(pv_parts, dim=0)
                cat_grid = torch.cat(grid_parts, dim=0)
                with torch.no_grad():
                    self.precomputed_image_out = base.get_image_features(cat_pv, cat_grid, return_dict=True)

    def kwargs_for_rows(self, row_indices: list[int] | None = None, *, allow_grad: bool = True) -> dict[str, Any]:
        if not self.has_multimodal:
            return {}
        indices = list(range(len(self.row_img_slices))) if row_indices is None else [int(i) for i in row_indices]
        img_indices: list[int] = []
        for b in indices:
            start, end = self.row_img_slices[b]
            img_indices.extend(range(start, end))
        if not img_indices:
            return {}
        if self.precomputed_image_out is not None and (not allow_grad or not self.vit_trainable):
            pooler = tuple(self.precomputed_image_out.pooler_output[i] for i in img_indices)
            deepstack = [
                tuple(layer_feats[i] for i in img_indices)
                for layer_feats in self.precomputed_image_out.deepstack_features
            ]
            return {"mm_encoder_outputs": {"image": SimpleNamespace(pooler_output=pooler, deepstack_features=deepstack)}}
        pv_parts = [self.per_row_pv[b] for b in indices if self.per_row_pv[b] is not None]
        grid_parts = [self.per_row_grid[b] for b in indices if self.per_row_grid[b] is not None]
        return {"pixel_values": torch.cat(pv_parts, dim=0), "image_grid_thw": torch.cat(grid_parts, dim=0)}


class Qwen3VLAdapter(StandardCausalAdapter):
    """Adapter for Qwen3-VL (e.g. Qwen3-VL-4B-Thinking) with DeepStack vision & 3D mRoPE."""

    name = "qwen3_vl"

    def __init__(self):
        # Qwen3VLTextModel calls create_causal_mask directly, which expects a raw 4D additive tensor.
        super().__init__(dict_mask=False)

    def freeze_vision(self, model) -> int:
        return freeze_vision_encoder(model)

    def lora_target_modules(self, model, *, freeze_vision: bool = True) -> str | list[str]:
        del model
        if freeze_vision:
            # Disjoint from Qwen3VLVisionModel linear names (qkv/proj/linear_fc1/linear_fc2).
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return "all-linear"

    def prepare_multimodal_batch(
        self, model, multimodal_inputs: list[dict | None] | None, device: torch.device
    ) -> MultimodalBatchHandle:
        return _Qwen3VLMultimodalBatch(model, multimodal_inputs, device)

    def build_position_ids(
        self,
        model,
        prompts: list[torch.Tensor],
        multimodal_inputs: list[dict | None] | None,
        base_position_ids: torch.Tensor,
        *,
        clean_lengths: list[int],
        canvas_start: int,
        noisy_start_offsets: list[int],
        noisy_lengths: list[int],
        pad_id: int,
    ) -> torch.Tensor:
        if not multimodal_inputs:
            return base_position_ids
        if not any(mm is not None and mm.get("image_grid_thw") is not None for mm in multimodal_inputs):
            return base_position_ids
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(base, "model", None)
        if inner is None or not hasattr(inner, "get_rope_index"):
            return base_position_ids

        device = base_position_ids.device
        batch_size, total = base_position_ids.shape
        pos_4d = base_position_ids[None, ...].expand(4, batch_size, total).clone()
        for b, mm in enumerate(multimodal_inputs):
            if mm is None or mm.get("image_grid_thw") is None:
                continue
            p = prompts[b].to(device=device, dtype=torch.long).view(-1)
            p_len = int(p.numel())
            p_valid = p != pad_id
            grid = mm["image_grid_thw"].to(device=device, dtype=torch.long)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            mm_type = _derive_qwen3_vl_token_types(p, base.config, mm)
            vision_pos, _ = inner.get_rope_index(
                p[None],
                mm_token_type_ids=mm_type[None],
                image_grid_thw=grid,
                attention_mask=p_valid[None],
            )
            pos_4d[1:, b, :p_len] = vision_pos[:, 0, :]
            next_pos = int(vision_pos[:, 0, p_valid].max().item()) + 1 if bool(p_valid.any()) else 0
            resp_prefix_len = max(0, int(clean_lengths[b]) - p_len)
            if resp_prefix_len > 0:
                pref_coords = torch.arange(resp_prefix_len, device=device, dtype=torch.long) + next_pos
                pos_4d[1:, b, p_len : p_len + resp_prefix_len] = pref_coords[None, :]
            n_len = int(noisy_lengths[b])
            if n_len > 0:
                n_coords = torch.arange(n_len, device=device, dtype=torch.long) + next_pos + int(noisy_start_offsets[b])
                pos_4d[1:, b, canvas_start : canvas_start + n_len] = n_coords[None, :]
        return pos_4d

    def build_cache_prefill_inputs(
        self,
        model,
        tokens_2d: torch.Tensor,
        valid_2d: torch.Tensor,
        multimodal_input: dict | None,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        device = tokens_2d.device
        seq_len = int(tokens_2d.shape[1])
        local_pos = (valid_2d.long().cumsum(dim=-1) - 1).clamp(min=0)
        next_text_pos = int(state.get("next_text_pos", 0))
        next_mrope_pos = int(state.get("next_mrope_pos", 0))
        use_4d_pos = bool(state.get("use_4d_pos", False))
        text_pos = local_pos + next_text_pos
        valid_count = int(valid_2d.sum().item())

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(base, "model", None)
        mm_kwargs: dict[str, Any] = {}
        if (
            multimodal_input is not None
            and multimodal_input.get("image_grid_thw") is not None
            and inner is not None
            and hasattr(inner, "get_rope_index")
        ):
            grid = multimodal_input["image_grid_thw"].to(device=device, dtype=torch.long)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            mm_type = _derive_qwen3_vl_token_types(tokens_2d[0], base.config, multimodal_input)
            vision_pos, _ = inner.get_rope_index(
                tokens_2d,
                mm_token_type_ids=mm_type[None],
                image_grid_thw=grid,
                attention_mask=valid_2d,
            )
            if next_mrope_pos > 0:
                vision_pos = vision_pos + next_mrope_pos
            pos = torch.cat([text_pos[None], vision_pos], dim=0)
            state["use_4d_pos"] = True
            state["next_mrope_pos"] = (
                int(vision_pos[:, 0, valid_2d[0]].max().item()) + 1 if valid_count > 0 else next_mrope_pos
            )
            if multimodal_input.get("pixel_values") is not None:
                mm_kwargs = {
                    "pixel_values": multimodal_input["pixel_values"].to(device=device),
                    "image_grid_thw": grid,
                }
        elif use_4d_pos:
            mrope_pos = (local_pos + next_mrope_pos)[None].expand(3, 1, seq_len)
            pos = torch.cat([text_pos[None], mrope_pos], dim=0)
            state["next_mrope_pos"] = next_mrope_pos + valid_count
        else:
            pos = text_pos
            state["next_mrope_pos"] = next_mrope_pos + valid_count
        state["next_text_pos"] = next_text_pos + valid_count
        return pos, mm_kwargs

    def build_cache_canvas_positions(
        self, canvas_len: int, device: torch.device, state: dict[str, Any]
    ) -> torch.Tensor:
        local_c = torch.arange(canvas_len, device=device, dtype=torch.long)[None, :]
        text_pos = local_c + int(state.get("next_text_pos", 0))
        if state.get("use_4d_pos"):
            mrope_pos = (local_c + int(state.get("next_mrope_pos", 0)))[None].expand(3, 1, canvas_len)
            return torch.cat([text_pos[None], mrope_pos], dim=0)
        return text_pos

    def load_backbone(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        processor_name: str | None = None,
    ) -> tuple[nn.Module, Any, Any]:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(processor_name or model_name)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to(device)
        return model, tokenizer, processor

    @contextmanager
    def forward_context(self, model, tokens: torch.Tensor, signal: torch.Tensor, canvas_start: int, conditioner: nn.Module):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = getattr(base, "model", None)
        embedding = _token_embedding(model)
        padding_idx = embedding.padding_idx
        placeholder_fn = getattr(inner, "get_placeholder_mask", None)
        had_placeholder = inner is not None and "get_placeholder_mask" in vars(inner)
        saved_placeholder = vars(inner).get("get_placeholder_mask") if inner is not None else None
        image_token_id = getattr(base.config, "image_token_id", None)
        video_token_id = getattr(base.config, "video_token_id", None)

        def embedding_hook(_module, _inputs, embeddings):
            canvas = embeddings[:, canvas_start:]
            conditioned = conditioner(canvas, signal.to(device=canvas.device, dtype=canvas.dtype))
            if canvas_start == 0:
                return conditioned
            return torch.cat([embeddings[:, :canvas_start], conditioned], dim=1)

        def guarded_placeholder_mask(_self, input_ids=None, inputs_embeds=None, image_features=None, video_features=None):
            # Prevent random noisy canvas tokens that accidentally collide with image_token_id/video_token_id
            # from breaking Qwen3VLModel's strict token-count vs visual-feature-count assertion.
            if input_ids is not None and canvas_start < input_ids.shape[1]:
                safe_ids = input_ids.clone()
                suffix = safe_ids[:, canvas_start:]
                replacement = 0
                if replacement in (image_token_id, video_token_id):
                    replacement = 1
                if image_token_id is not None:
                    suffix = torch.where(suffix == int(image_token_id), torch.full_like(suffix, replacement), suffix)
                if video_token_id is not None:
                    suffix = torch.where(suffix == int(video_token_id), torch.full_like(suffix, replacement), suffix)
                safe_ids[:, canvas_start:] = suffix
                return placeholder_fn(
                    safe_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_features,
                    video_features=video_features,
                )
            return placeholder_fn(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )

        handle = embedding.register_forward_hook(embedding_hook)
        try:
            embedding.padding_idx = None
            if placeholder_fn is not None:
                inner.get_placeholder_mask = MethodType(guarded_placeholder_mask, inner)
            yield
        finally:
            handle.remove()
            embedding.padding_idx = padding_idx
            if placeholder_fn is not None:
                if had_placeholder:
                    inner.get_placeholder_mask = saved_placeholder
                else:
                    delattr(inner, "get_placeholder_mask")


def freeze_vision_encoder(model) -> int:
    """Freeze vision tower parameters (if present) and put it in eval mode. Returns frozen param count."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = getattr(getattr(base, "model", None), "visual", None) or getattr(base, "visual", None)
    if visual is None:
        return 0
    frozen = 0
    for param in visual.parameters():
        param.requires_grad_(False)
        frozen += param.numel()
    visual.eval()
    return frozen


_ADAPTER_REGISTRY: dict[str, DiffusionModelAdapter] = {
    "gemma4": Gemma4Adapter(),
    "gemma4_text": Gemma4Adapter(),
    "gemma3": Gemma4Adapter(),
    "gemma3_text": Gemma4Adapter(),
    "qwen3": StandardCausalAdapter(dict_mask=True),
    "qwen2": StandardCausalAdapter(dict_mask=True),
    "qwen3_vl": Qwen3VLAdapter(),
    "qwen3_vl_text": Qwen3VLAdapter(),
    "llama": StandardCausalAdapter(dict_mask=False),
    "mistral": StandardCausalAdapter(dict_mask=False),
}


def register_diffusion_adapter(model_type: str, adapter: DiffusionModelAdapter) -> None:
    """Register a custom ``DiffusionModelAdapter`` for a Hugging Face ``model_type``."""
    _ADAPTER_REGISTRY[model_type.lower()] = adapter


def get_diffusion_adapter(model) -> DiffusionModelAdapter:
    """Resolve the ``DiffusionModelAdapter`` for ``model`` without hardcoding logic in loss.py."""
    config = getattr(model, "config", model)
    text_config = getattr(config, "text_config", config)
    model_type = str(getattr(text_config, "model_type", "") or getattr(config, "model_type", "")).lower()
    if model_type in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[model_type]
    return StandardCausalAdapter()


def load_diffusion_backbone(
    model_name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    processor_name: str | None = None,
) -> tuple[nn.Module, Any, Any, DiffusionModelAdapter]:
    """Resolve adapter from HF config and load ``(model, tokenizer, processor, adapter)``."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    adapter = get_diffusion_adapter(cfg)
    model, tokenizer, processor = adapter.load_backbone(
        model_name, dtype=dtype, device=device, processor_name=processor_name
    )
    return model, tokenizer, processor, adapter
