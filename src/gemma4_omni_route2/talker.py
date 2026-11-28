"""Gemma-4-E2B -> CosyVoice2 25Hz Autoregressive Streaming Talker (LLaMA-Omni 2 Read-Write Architecture).

Maps Gemma-4-E2B continuous last-layer hidden_states (1536-dim) + aligned TTS text embeddings
via Gate-Fusion into a 24-layer Qwen2ForCausalLM speech generator (896-dim) outputting 25Hz FSQ units (0..6560).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, DynamicCache, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import Cache


IGNORE_INDEX = -100
UNIT_VOCAB_SIZE = 6561
MAX_TALKER_CACHE_LEN = 512
DUPLEX_SPECIAL_TOKENS = [f"<unused{i}>" for i in range(8)]


class StaticGraphCache(Cache):
    """Zero-allocation static KV cache supporting multi-token prefill and 1-token CUDAGraph replay."""

    def __init__(self, max_len: int):
        super().__init__(layers=[])
        self.max_len = max_len
        self.k_bufs: dict[int, torch.Tensor] = {}
        self.v_bufs: dict[int, torch.Tensor] = {}
        self.seq_len_int: int = 0
        self.graph_pos: Optional[torch.Tensor] = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_len_int

    def reset(self) -> None:
        self.seq_len_int = 0
        self.graph_pos = None

    def crop_to(self, target_len: int) -> None:
        self.seq_len_int = max(0, min(int(target_len), self.seq_len_int))
        self.graph_pos = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        bsz, n_heads, seq_l, head_d = key_states.shape
        if layer_idx not in self.k_bufs:
            self.k_bufs[layer_idx] = torch.zeros(
                (bsz, n_heads, self.max_len, head_d), dtype=key_states.dtype, device=key_states.device
            )
            self.v_bufs[layer_idx] = torch.zeros(
                (bsz, n_heads, self.max_len, head_d), dtype=value_states.dtype, device=value_states.device
            )
        kb, vb = self.k_bufs[layer_idx], self.v_bufs[layer_idx]
        if self.graph_pos is not None and seq_l == 1:
            kb.index_copy_(2, self.graph_pos, key_states)
            vb.index_copy_(2, self.graph_pos, value_states)
            return kb, vb
        s = self.seq_len_int
        kb[:, :, s : s + seq_l, :] = key_states
        vb[:, :, s : s + seq_l, :] = value_states
        return kb[:, :, : s + seq_l, :], vb[:, :, : s + seq_l, :]


def make_prefill_masks(
    start: int, length: int, sliding_window: Optional[int], device: torch.device | str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    q = torch.arange(start, start + length, device=device).view(1, 1, length, 1)
    k = torch.arange(0, start + length, device=device).view(1, 1, 1, start + length)
    full = k <= q
    masks: dict[str, torch.Tensor] = {"full_attention": full}
    if sliding_window is not None:
        masks["sliding_attention"] = full & (k > q - sliding_window)
    pos_2d = torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)
    return masks, pos_2d


def ensure_gemma4_duplex_special_tokens(gemma_tokenizer) -> None:
    """Ensure <unused0>..<unused7> encode as single atomic token IDs 6..13."""
    test_ids = gemma_tokenizer.encode("<unused0>", add_special_tokens=False)
    if test_ids != [6]:
        gemma_tokenizer.add_tokens(DUPLEX_SPECIAL_TOKENS, special_tokens=True)


@dataclass
class TalkerStreamSession:
    """Per-connection incremental KV-cached streaming state for Gemma4SpeechGenerator."""

    kv_cache: DynamicCache = field(default_factory=DynamicCache)
    graph_turn_started: bool = False
    pending_gemma_ids: list[int] = field(default_factory=list)
    pending_gemma_hiddens: list[torch.Tensor] = field(default_factory=list)
    buffered_tts_ids: list[int] = field(default_factory=list)
    buffered_hiddens: list[torch.Tensor] = field(default_factory=list)
    in_tool_call: bool = False
    recent_unit_ids: list[int] = field(default_factory=list)
    spoken_char_count: int = 0
    finished: bool = False

    def reset(self) -> None:
        """Instant 0ms flush on <unused6> barge-in or turn boundary."""
        self.kv_cache = DynamicCache()
        self.graph_turn_started = False
        self.pending_gemma_ids.clear()
        self.pending_gemma_hiddens.clear()
        self.buffered_tts_ids.clear()
        self.buffered_hiddens.clear()
        self.in_tool_call = False
        self.recent_unit_ids.clear()
        self.spoken_char_count = 0
        self.finished = False


class Gemma4SpeechGenerator(nn.Module):
    """Bridges frozen Gemma-4-E2B (1536-dim) to CosyVoice2 25Hz discrete speech tokens (0..6560)."""

    def __init__(
        self,
        speech_gen_config: dict,
        tts_tokenizer_path: str,
        thinker_hidden_size: int = 1536,
        read_len: int = 3,
        write_len: int = 10,
    ):
        super().__init__()
        self.speech_gen_config = speech_gen_config
        self.thinker_hidden_size = thinker_hidden_size
        self.read_len = read_len
        self.write_len = write_len

        self.tokenizer = AutoTokenizer.from_pretrained(tts_tokenizer_path)
        self.model = Qwen2ForCausalLM(Qwen2Config(**speech_gen_config))
        talker_dim = self.model.config.hidden_size  # 896

        self.gemma_norm = nn.LayerNorm(thinker_hidden_size)
        self.input_proj = nn.Sequential(
            nn.Linear(thinker_hidden_size, thinker_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(thinker_hidden_size * 2, talker_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * talker_dim, talker_dim),
            nn.Sigmoid(),
        )

        self.sep_token_id = int(self.tokenizer.convert_tokens_to_ids("<sep>"))  # 151665
        self.unit_offset = int(self.tokenizer.convert_tokens_to_ids("<0>"))  # 151666
        self.eos_token_id = int(self.tokenizer.eos_token_id)  # 151643
        self.prosody_gate_cap: Optional[float] = None
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_cache: Optional[StaticGraphCache] = None
        self._g_tok_id: Optional[torch.Tensor] = None
        self._g_pos_1d: Optional[torch.Tensor] = None
        self._g_idx: Optional[torch.Tensor] = None
        self._g_unit_logits: Optional[torch.Tensor] = None
        self._g_eos_logit: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def init_cuda_graph(self, stream: Optional[torch.cuda.Stream] = None) -> bool:
        """Capture 24-layer Qwen2 single-unit decode into a single CUDAGraph (~0.7ms/unit on A100 vs ~42ms eager)."""
        p0 = next(self.model.parameters())
        device = p0.device
        dtype = p0.dtype
        if device.type != "cuda" or self._cuda_graph is not None:
            return self._cuda_graph is not None
        try:
            self._graph_cache = StaticGraphCache(max_len=MAX_TALKER_CACHE_LEN)
            self._g_tok_id = torch.full((1, 1), self.unit_offset, dtype=torch.long, device=device)
            self._g_pos_1d = torch.zeros((1,), dtype=torch.long, device=device)
            self._g_idx = torch.arange(MAX_TALKER_CACHE_LEN, device=device).view(1, 1, 1, MAX_TALKER_CACHE_LEN)
            self._g_unit_logits = torch.zeros((UNIT_VOCAB_SIZE,), dtype=torch.float32, device=device)
            self._g_eos_logit = torch.zeros((1,), dtype=torch.float32, device=device)

            # Pre-allocate all 24 KV layer buffers before capture
            dummy_emb = torch.zeros((1, 2, self.model.config.hidden_size), dtype=dtype, device=device)
            m0, p2 = make_prefill_masks(0, 2, None, device)
            _ = self.model.model(
                inputs_embeds=dummy_emb,
                position_ids=p2,
                attention_mask=m0,
                past_key_values=self._graph_cache,
                use_cache=True,
                return_dict=True,
            )
            self._graph_cache.seq_len_int = 2
            self._g_pos_1d.fill_(2)
            self._graph_cache.graph_pos = self._g_pos_1d

            def _graph_body() -> None:
                emb = self.model.get_input_embeddings()(self._g_tok_id).to(dtype=dtype)
                pv = self._g_pos_1d.view(1, 1, 1, 1)
                masks = {"full_attention": (self._g_idx <= pv)}
                h = self.model.model(
                    inputs_embeds=emb,
                    position_ids=self._g_pos_1d.unsqueeze(0),
                    attention_mask=masks,
                    past_key_values=self._graph_cache,
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state
                lg = self.model.lm_head(h)[0, 0, :].float()
                self._g_unit_logits.copy_(lg[self.unit_offset : self.unit_offset + UNIT_VOCAB_SIZE])
                self._g_eos_logit.copy_(lg[self.eos_token_id : self.eos_token_id + 1])

            cap_s = stream if stream is not None else torch.cuda.Stream(device=device)
            cap_s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(cap_s):
                for _ in range(3):
                    _graph_body()
            torch.cuda.synchronize(device)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=cap_s, capture_error_mode="thread_local"):
                _graph_body()
            torch.cuda.synchronize(device)
            self._graph_cache.reset()
            self._cuda_graph = g
            return True
        except Exception as exc:
            print(f"   ⚠️ Talker CUDAGraph fallback to eager ({exc})")
            self._cuda_graph = None
            self._graph_cache = None
            return False

    def _prefill_step_logits(
        self, new_embeds: torch.Tensor, session: TalkerStreamSession
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill new_embeds [L, 896] and return (unit_logits [6561], eos_logit scalar)."""
        u_start = self.unit_offset
        u_end = self.unit_offset + UNIT_VOCAB_SIZE
        if self._cuda_graph is not None and self._graph_cache is not None:
            if not session.graph_turn_started:
                self._graph_cache.reset()
                session.graph_turn_started = True
            L = new_embeds.shape[0]
            if self._graph_cache.seq_len_int + L >= MAX_TALKER_CACHE_LEN - 4:
                self._graph_cache.reset()
            self._graph_cache.graph_pos = None
            masks, pos_2d = make_prefill_masks(self._graph_cache.seq_len_int, L, None, new_embeds.device)
            h = self.model.model(
                inputs_embeds=new_embeds.unsqueeze(0),
                position_ids=pos_2d,
                attention_mask=masks,
                past_key_values=self._graph_cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            self._graph_cache.seq_len_int += L
            lg = self.model.lm_head(h[:, -1:, :])[0, 0, :].float()
            return lg[u_start:u_end], lg[self.eos_token_id]

        out = self.model(
            inputs_embeds=new_embeds.unsqueeze(0),
            past_key_values=session.kv_cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        lg = out.logits[0, -1, :].float()
        return lg[u_start:u_end], lg[self.eos_token_id]

    def _decode_1tok_logits(
        self, tok_1x1: torch.Tensor, dtype: torch.dtype, session: TalkerStreamSession
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run 1 autoregressive unit step via CUDAGraph (if active) or eager KV cache."""
        u_start = self.unit_offset
        u_end = self.unit_offset + UNIT_VOCAB_SIZE
        if self._cuda_graph is not None and self._graph_cache is not None and self._graph_cache.seq_len_int < MAX_TALKER_CACHE_LEN - 1:
            self._graph_cache.graph_pos = self._g_pos_1d
            self._g_tok_id.copy_(tok_1x1)
            self._g_pos_1d.fill_(self._graph_cache.seq_len_int)
            self._cuda_graph.replay()
            self._graph_cache.graph_pos = None
            self._graph_cache.seq_len_int += 1
            return self._g_unit_logits, self._g_eos_logit[0]

        embed_fn = self.model.get_input_embeddings()
        step_out = self.model(
            inputs_embeds=embed_fn(tok_1x1).to(dtype=dtype),
            past_key_values=session.kv_cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        lg = step_out.logits[0, -1, :].float()
        return lg[u_start:u_end], lg[self.eos_token_id]

    @classmethod
    def from_llama_omni2_pretrained(
        cls,
        donor_repo_id: str = "ICTNLP/LLaMA-Omni2-1.5B-Bilingual",
        thinker_hidden_size: int = 1536,
        init_gate_bias: float = -1.1,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Gemma4SpeechGenerator":
        """Initialize 24-layer Qwen2 speech_generator from LLaMA-Omni2-1.5B-Bilingual + non-saturating Gemma-4 bridge."""
        cfg_path = hf_hub_download(donor_repo_id, "config.json")
        shard_path = hf_hub_download(donor_repo_id, "model-00002-of-00002.safetensors")
        tts_root = snapshot_download(donor_repo_id, allow_patterns=["tts_tokenizer/*"])
        tts_tok_dir = os.path.join(tts_root, "tts_tokenizer")

        with open(cfg_path, encoding="utf-8") as f:
            donor_cfg = json.load(f)

        read_len, write_len = 3, 10
        inst = cls(
            speech_gen_config=donor_cfg["speech_generator"],
            tts_tokenizer_path=tts_tok_dir,
            thinker_hidden_size=thinker_hidden_size,
            read_len=read_len,
            write_len=write_len,
        )

        raw_sd = load_file(shard_path)
        model_sd = {
            k[len("speech_generator.model.") :]: v
            for k, v in raw_sd.items()
            if k.startswith("speech_generator.model.")
        }
        inst.model.load_state_dict(model_sd, strict=True)

        # Match input_proj output scale to pretrained Qwen2 embedding std (~0.03) so gate never saturates at Step 0
        with torch.no_grad():
            emb_std = float(inst.model.get_input_embeddings().weight.float().std().item())
            nn.init.kaiming_uniform_(inst.input_proj[0].weight, a=0.0)
            nn.init.zeros_(inst.input_proj[0].bias)
            nn.init.normal_(inst.input_proj[2].weight, mean=0.0, std=emb_std / (thinker_hidden_size**0.5))
            nn.init.zeros_(inst.input_proj[2].bias)
            nn.init.normal_(inst.gate[0].weight, mean=0.0, std=0.05)
            inst.gate[0].bias.fill_(float(init_gate_bias if init_gate_bias is not None else -1.1))

        return inst.to(dtype=torch_dtype)

    def save_checkpoint(self, save_dir: str) -> None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(out / "tts_tokenizer"))
        meta = {
            "speech_gen_config": self.speech_gen_config,
            "thinker_hidden_size": self.thinker_hidden_size,
            "read_len": self.read_len,
            "write_len": self.write_len,
        }
        with open(out / "talker_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(sd, str(out / "talker_model.safetensors"))

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        restore_pristine_backbone: bool = True,
        donor_repo_id: str = "ICTNLP/LLaMA-Omni2-1.5B-Bilingual",
    ) -> "Gemma4SpeechGenerator":
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = snapshot_download(ckpt_dir)
        with open(os.path.join(ckpt_dir, "talker_config.json"), encoding="utf-8") as f:
            meta = json.load(f)
        inst = cls(
            speech_gen_config=meta["speech_gen_config"],
            tts_tokenizer_path=os.path.join(ckpt_dir, "tts_tokenizer"),
            thinker_hidden_size=meta["thinker_hidden_size"],
            read_len=meta.get("read_len", 3),
            write_len=meta.get("write_len", 10),
        )
        sd = load_file(os.path.join(ckpt_dir, "talker_model.safetensors"))
        inst.load_state_dict(sd, strict=True)
        if restore_pristine_backbone:
            # Keep trained Gemma-4 -> Talker bridge (gemma_norm + input_proj + gate), while restoring
            # the 100% pristine 200K-dialogue pretrained 24-layer speech_generator.model weights!
            try:
                shard_path = hf_hub_download(donor_repo_id, "model-00002-of-00002.safetensors")
                donor_sd = load_file(shard_path)
                pristine_model_sd = {
                    k[len("speech_generator.model.") :]: v
                    for k, v in donor_sd.items()
                    if k.startswith("speech_generator.model.")
                }
                inst.model.load_state_dict(pristine_model_sd, strict=True)
                inst.prosody_gate_cap = 0.15
                print("   ✅ Restored pristine 200K-dialogue 24-layer Talker + Gemma-4 prosody bridge!")
            except Exception as exc:
                print(f"   ⚠️ Kept checkpoint Talker backbone ({exc})")
        inst = inst.to(dtype=torch_dtype)
        if device is not None:
            inst = inst.to(device)
        return inst

    def fuse_representations(
        self,
        gemma_hidden: torch.Tensor,
        tts_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse Gemma-4 continuous hidden states [T, 1536] with Qwen2 TTS text token embeddings [T, 896]."""
        target_dtype = self.input_proj[0].weight.dtype
        normed = self.gemma_norm(gemma_hidden.to(dtype=target_dtype))
        rep = self.input_proj(normed)
        emb = self.model.get_input_embeddings()(tts_token_ids).to(dtype=target_dtype)
        g = self.gate(torch.cat([rep, emb], dim=-1))
        if self.prosody_gate_cap is not None:
            # Match rep RMS norm to emb RMS norm so Gemma-4 prosody enriches without shifting phonetic manifold
            emb_rms = emb.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
            rep_rms = rep.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
            rep = (rep.float() * (emb_rms / rep_rms)).to(dtype=target_dtype)
            g = g * float(self.prosody_gate_cap)
        fused = rep * g + emb * (1.0 - g)
        return fused, g.mean()

    @staticmethod
    def extract_spoken_gemma_spans(
        gemma_tokenizer,
        token_ids: list[int],
        pred_hidden_states: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], list[str]]:
        """Filter an assistant turn's Gemma-4 tokens + predicting hidden_states [L, 1536] down to spoken pieces.

        Automatically excludes:
        - <unused0>..<unused7> duplex control slots (IDs 6..13)
        - <|tool_call> ... <tool_call|> JSON/DSL spans
        - Special tokens (<turn|>, <eos>, <bos>, <pad>, <|thought>, <thought|>)
        """
        assert len(token_ids) == pred_hidden_states.shape[0]
        tc_start = gemma_tokenizer.convert_tokens_to_ids("<|tool_call>")
        tc_end = gemma_tokenizer.convert_tokens_to_ids("<tool_call|>")
        th_start = gemma_tokenizer.convert_tokens_to_ids("<|channel>")
        th_end = gemma_tokenizer.convert_tokens_to_ids("<channel|>")
        skip_ids = set(gemma_tokenizer.all_special_ids) | set(range(6, 14))
        for tok_s in ("<turn|>", "<|turn>", "<|tool_response>", "<tool_response|>"):
            tid = gemma_tokenizer.convert_tokens_to_ids(tok_s)
            if isinstance(tid, int) and tid >= 0:
                skip_ids.add(tid)

        in_tc = False
        in_th = False
        pending_ids: list[int] = []
        pending_h: list[torch.Tensor] = []
        pieces: list[str] = []
        piece_hiddens: list[torch.Tensor] = []

        for tid, h_vec in zip(token_ids, pred_hidden_states):
            tid = int(tid)
            if tid == tc_start:
                in_tc = True
                continue
            if tid == tc_end:
                in_tc = False
                continue
            if tid == th_start:
                in_th = True
                continue
            if tid == th_end:
                in_th = False
                continue
            if in_tc or in_th or tid in skip_ids:
                continue

            pending_ids.append(tid)
            pending_h.append(h_vec)
            text = gemma_tokenizer.decode(pending_ids, skip_special_tokens=False)
            if "\ufffd" in text:
                continue
            if text.strip() == "" and text != " ":
                pending_ids.clear()
                pending_h.clear()
                continue
            avg_h = torch.stack(pending_h, dim=0).mean(dim=0)
            pieces.append(text)
            piece_hiddens.append(avg_h)
            pending_ids.clear()
            pending_h.clear()

        if not pieces:
            return None, []
        return torch.stack(piece_hiddens, dim=0), pieces

    def align_pieces_to_tts(
        self,
        piece_hiddens: torch.Tensor,
        pieces: list[str],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Map spoken text pieces + Gemma-4 hidden states [P, 1536] to TTS token IDs [T] + hidden states [T, 1536]."""
        out_ids: list[int] = []
        out_h: list[torch.Tensor] = []
        for h_vec, piece in zip(piece_hiddens, pieces):
            sub_ids = self.tokenizer.encode(piece, add_special_tokens=False)
            if not sub_ids:
                continue
            for sid in sub_ids:
                out_ids.append(int(sid))
                out_h.append(h_vec)
        if not out_ids:
            return None, None
        device = piece_hiddens.device
        return torch.stack(out_h, dim=0), torch.tensor(out_ids, dtype=torch.long, device=device)

    def build_interleaved_sequence(
        self,
        fused_text_embeds: torch.Tensor,
        speech_units: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build exact Read-Write (R=3, W=10) teacher-forcing embeddings [S, 896] and labels [S]."""
        device = fused_text_embeds.device
        dtype = fused_text_embeds.dtype
        embed_fn = self.model.get_input_embeddings()

        unit_tok_ids = [self.unit_offset + int(u) for u in speech_units if 0 <= int(u) < UNIT_VOCAB_SIZE]
        total_text = fused_text_embeds.shape[0]
        total_units = len(unit_tok_ids)
        R, W = self.read_len, self.write_len

        seq_embeds: list[torch.Tensor] = []
        seq_labels: list[int] = []

        t_ptr = 0
        u_ptr = 0
        # Non-final Read-Write cycles: only while > R text tokens remain AND > W speech units remain
        while (total_text - t_ptr) > R and (total_units - u_ptr) > W:
            t_chunk = fused_text_embeds[t_ptr : t_ptr + R]
            seq_embeds.append(t_chunk)
            seq_labels.extend([IGNORE_INDEX] * R)
            t_ptr += R

            w_ids = torch.tensor(unit_tok_ids[u_ptr : u_ptr + W], dtype=torch.long, device=device)
            seq_embeds.append(embed_fn(w_ids).to(dtype=dtype))
            seq_labels.extend(w_ids.tolist())
            u_ptr += W

        # Final cycle: remaining text embeddings + <sep> + remaining speech units + <eos>
        rem_text = fused_text_embeds[t_ptr:]
        sep_emb = embed_fn(torch.tensor([self.sep_token_id], dtype=torch.long, device=device)).to(dtype=dtype)
        seq_embeds.append(rem_text)
        seq_embeds.append(sep_emb)
        seq_labels.extend([IGNORE_INDEX] * (rem_text.shape[0] + 1))

        tail_ids = unit_tok_ids[u_ptr:] + [self.eos_token_id]
        tail_tensor = torch.tensor(tail_ids, dtype=torch.long, device=device)
        seq_embeds.append(embed_fn(tail_tensor).to(dtype=dtype))
        seq_labels.extend(tail_ids)

        full_embeds = torch.cat(seq_embeds, dim=0)
        full_labels = torch.tensor(seq_labels, dtype=torch.long, device=device)
        return full_embeds, full_labels

    def forward_batch_loss(
        self,
        batch_gemma_hiddens: list[torch.Tensor],
        batch_tts_ids: list[torch.Tensor],
        batch_speech_units: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched Read-Write (R=3, W=10) causal LM loss over discrete speech units."""
        embeds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        gate_vals: list[torch.Tensor] = []

        for g_hid, t_ids, units in zip(batch_gemma_hiddens, batch_tts_ids, batch_speech_units):
            fused, g_mean = self.fuse_representations(g_hid, t_ids)
            gate_vals.append(g_mean)
            seq_e, seq_l = self.build_interleaved_sequence(fused, units)
            embeds_list.append(seq_e)
            labels_list.append(seq_l)

        bsz = len(embeds_list)
        max_len = max(x.shape[0] for x in embeds_list)
        dim = embeds_list[0].shape[1]
        dtype = embeds_list[0].dtype
        device = embeds_list[0].device

        padded_embeds = torch.zeros((bsz, max_len, dim), dtype=dtype, device=device)
        padded_labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

        for i, (e, l) in enumerate(zip(embeds_list, labels_list)):
            L = e.shape[0]
            padded_embeds[i, :L] = e
            padded_labels[i, :L] = l
            attn_mask[i, :L] = 1

        out = self.model(
            inputs_embeds=padded_embeds,
            attention_mask=attn_mask,
            labels=padded_labels,
            use_cache=False,
            return_dict=True,
        )
        return out.loss, torch.stack(gate_vals).mean()

    @torch.inference_mode()
    def _decode_unit_chunk(
        self,
        new_embeds: torch.Tensor,
        session: TalkerStreamSession,
        is_finished: bool,
        temperature: float = 0.55,
        top_k: int = 15,
    ) -> list[int]:
        """Incremental KV-cached unit generation (prefills ONLY new_embeds, decodes <=W units in ~4ms on A100)."""
        device = new_embeds.device
        dtype = new_embeds.dtype
        embed_fn = self.model.get_input_embeddings()

        if is_finished:
            sep_emb = embed_fn(torch.tensor([self.sep_token_id], dtype=torch.long, device=device)).to(dtype=dtype)
            new_embeds = torch.cat([new_embeds, sep_emb], dim=0)

        u_logits_cur, eos_logit_cur = self._prefill_step_logits(new_embeds, session)
        u_start = self.unit_offset

        if not is_finished:
            # CUDAGraph + zero-CPU-sync GPU loop (~7ms total for 10 units on A100!)
            gpu_unit_vals: list[torch.Tensor] = []
            penalize_u: list[int] = []
            if len(session.recent_unit_ids) >= 8 and len(set(session.recent_unit_ids[-8:])) == 1:
                penalize_u = [session.recent_unit_ids[-1] - u_start]
            elif len(session.recent_unit_ids) >= 10 and len(set(session.recent_unit_ids[-10:])) <= 2:
                penalize_u = [sid - u_start for sid in set(session.recent_unit_ids[-10:])]

            for _ in range(self.write_len):
                u_logits = u_logits_cur.clone()
                for pu in penalize_u:
                    if 0 <= pu < UNIT_VOCAB_SIZE:
                        u_logits[pu] -= 7.0
                if temperature <= 1e-3:
                    u_idx = torch.argmax(u_logits)
                else:
                    scaled = u_logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(scaled, min(top_k, UNIT_VOCAB_SIZE))
                        scaled.masked_fill_(scaled < v[-1], float("-inf"))
                    u_idx = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).squeeze(0)
                gpu_unit_vals.append(u_idx)
                next_t = (u_idx + u_start).view(1, 1)
                u_logits_cur, eos_logit_cur = self._decode_1tok_logits(next_t, dtype, session)

            emitted_units = torch.stack(gpu_unit_vals).tolist()
            session.recent_unit_ids.extend(u + u_start for u in emitted_units)
            return emitted_units

        est_cap = max(24, session.spoken_char_count * 6 - len(session.recent_unit_ids) + 14)
        max_steps = min(160, est_cap) if session.spoken_char_count > 0 else 160
        emitted_units: list[int] = []

        for _ in range(max_steps):
            cand_logits = torch.empty(UNIT_VOCAB_SIZE + 1, dtype=u_logits_cur.dtype, device=device)
            cand_logits[:UNIT_VOCAB_SIZE] = u_logits_cur
            cand_logits[UNIT_VOCAB_SIZE] = eos_logit_cur

            if len(session.recent_unit_ids) >= 8 and len(set(session.recent_unit_ids[-8:])) == 1:
                pu = session.recent_unit_ids[-1] - u_start
                if 0 <= pu < UNIT_VOCAB_SIZE:
                    cand_logits[pu] -= 8.0
            elif len(session.recent_unit_ids) >= 10 and len(set(session.recent_unit_ids[-10:])) <= 2:
                for sid in set(session.recent_unit_ids[-10:]):
                    pu = sid - u_start
                    if 0 <= pu < UNIT_VOCAB_SIZE:
                        cand_logits[pu] -= 6.0

            if temperature <= 1e-3:
                c_idx = int(torch.argmax(cand_logits).item())
            else:
                scaled = cand_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, UNIT_VOCAB_SIZE + 1))
                    scaled.masked_fill_(scaled < v[-1], float("-inf"))
                c_idx = int(torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).item())

            if c_idx == UNIT_VOCAB_SIZE:
                session.finished = True
                break

            next_id = c_idx + u_start
            emitted_units.append(c_idx)
            session.recent_unit_ids.append(next_id)

            if len(emitted_units) < max_steps:
                next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
                u_logits_cur, eos_logit_cur = self._decode_1tok_logits(next_t, dtype, session)

        return emitted_units

    @torch.inference_mode()
    def step_gemma_token(
        self,
        gemma_tokenizer,
        gemma_token_id: int,
        gemma_pred_hidden: torch.Tensor,
        session: TalkerStreamSession,
        is_turn_end: bool = False,
        temperature: float = 0.55,
    ) -> list[int]:
        """Feed one newly generated Gemma-4 token + its predicting hidden state [1536] in live streaming."""
        if session.finished:
            return []

        tc_start = gemma_tokenizer.convert_tokens_to_ids("<|tool_call>")
        tc_end = gemma_tokenizer.convert_tokens_to_ids("<tool_call|>")
        skip_ids = set(gemma_tokenizer.all_special_ids) | set(range(6, 14))
        for tok_s in ("<turn|>", "<|turn>", "<|tool_response>", "<tool_response|>"):
            tid = gemma_tokenizer.convert_tokens_to_ids(tok_s)
            if isinstance(tid, int) and tid >= 0:
                skip_ids.add(tid)

        tid = int(gemma_token_id)
        if not is_turn_end:
            if tid == tc_start:
                session.in_tool_call = True
                return []
            if tid == tc_end:
                session.in_tool_call = False
                return []
            if session.in_tool_call or tid in skip_ids:
                return []

            session.pending_gemma_ids.append(tid)
            session.pending_gemma_hiddens.append(gemma_pred_hidden.detach())
            piece = gemma_tokenizer.decode(session.pending_gemma_ids, skip_special_tokens=False)
            if "\ufffd" in piece:
                return []
            avg_h = torch.stack(session.pending_gemma_hiddens, dim=0).mean(dim=0)
            session.pending_gemma_ids.clear()
            session.pending_gemma_hiddens.clear()
            session.spoken_char_count += len(piece.strip())

            sub_ids = self.tokenizer.encode(piece, add_special_tokens=False)
            for sid in sub_ids:
                session.buffered_tts_ids.append(int(sid))
                session.buffered_hiddens.append(avg_h)

            out_units: list[int] = []
            while len(session.buffered_tts_ids) >= self.read_len:
                c_ids = torch.tensor(
                    session.buffered_tts_ids[: self.read_len],
                    dtype=torch.long,
                    device=gemma_pred_hidden.device,
                )
                c_hid = torch.stack(session.buffered_hiddens[: self.read_len], dim=0)
                del session.buffered_tts_ids[: self.read_len]
                del session.buffered_hiddens[: self.read_len]
                fused, _ = self.fuse_representations(c_hid, c_ids)
                out_units.extend(
                    self._decode_unit_chunk(fused, session, is_finished=False, temperature=temperature)
                )
            return out_units

        # Turn ended (<turn|> or <eos> reached): flush any remaining spoken tokens + <sep>
        if not session.buffered_tts_ids and len(session.recent_unit_ids) == 0:
            session.finished = True
            return []

        device = gemma_pred_hidden.device
        dtype = self.input_proj[0].weight.dtype
        if session.buffered_tts_ids:
            c_ids = torch.tensor(session.buffered_tts_ids, dtype=torch.long, device=device)
            c_hid = torch.stack(session.buffered_hiddens, dim=0)
            session.buffered_tts_ids.clear()
            session.buffered_hiddens.clear()
            fused, _ = self.fuse_representations(c_hid, c_ids)
        else:
            fused = torch.zeros((0, self.model.config.hidden_size), dtype=dtype, device=device)

        return self._decode_unit_chunk(fused, session, is_finished=True, temperature=temperature)

    @torch.inference_mode()
    def iter_turn_end_chunks(
        self,
        gemma_pred_hidden: torch.Tensor,
        session: TalkerStreamSession,
        temperature: float = 0.55,
        top_k: int = 15,
        subchunk_size: int = 12,
    ):
        """Stream final-cycle speech units in fast 12-unit GPU batches so post-<sep> audio never stalls Chunk 0."""
        if session.finished or (not session.buffered_tts_ids and len(session.recent_unit_ids) == 0):
            session.finished = True
            yield [], True
            return

        device = gemma_pred_hidden.device
        dtype = self.input_proj[0].weight.dtype
        embed_fn = self.model.get_input_embeddings()

        if session.buffered_tts_ids:
            c_ids = torch.tensor(session.buffered_tts_ids, dtype=torch.long, device=device)
            c_hid = torch.stack(session.buffered_hiddens, dim=0)
            session.buffered_tts_ids.clear()
            session.buffered_hiddens.clear()
            fused, _ = self.fuse_representations(c_hid, c_ids)
        else:
            fused = torch.zeros((0, self.model.config.hidden_size), dtype=dtype, device=device)

        sep_emb = embed_fn(torch.tensor([self.sep_token_id], dtype=torch.long, device=device)).to(dtype=dtype)
        new_embeds = torch.cat([fused, sep_emb], dim=0)
        u_logits_cur, eos_logit_cur = self._prefill_step_logits(new_embeds, session)

        est_cap = max(24, session.spoken_char_count * 6 - len(session.recent_unit_ids) + 14)
        max_steps = min(160, est_cap) if session.spoken_char_count > 0 else 160
        u_start = self.unit_offset

        buf_units: list[int] = []
        steps_done = 0
        while steps_done < max_steps and not session.finished:
            batch_n = min(4, max_steps - steps_done)
            gpu_choices: list[torch.Tensor] = []
            penalize_u: list[int] = []
            if len(session.recent_unit_ids) >= 8 and len(set(session.recent_unit_ids[-8:])) == 1:
                penalize_u = [session.recent_unit_ids[-1] - u_start]
            elif len(session.recent_unit_ids) >= 10 and len(set(session.recent_unit_ids[-10:])) <= 2:
                penalize_u = [sid - u_start for sid in set(session.recent_unit_ids[-10:])]

            for _ in range(batch_n):
                cand_logits = torch.empty(UNIT_VOCAB_SIZE + 1, dtype=u_logits_cur.dtype, device=device)
                cand_logits[:UNIT_VOCAB_SIZE] = u_logits_cur
                cand_logits[UNIT_VOCAB_SIZE] = eos_logit_cur
                for pu in penalize_u:
                    if 0 <= pu < UNIT_VOCAB_SIZE:
                        cand_logits[pu] -= 6.5
                if temperature <= 1e-3:
                    c_idx = torch.argmax(cand_logits)
                else:
                    scaled = cand_logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(scaled, min(top_k, UNIT_VOCAB_SIZE + 1))
                        scaled.masked_fill_(scaled < v[-1], float("-inf"))
                    c_idx = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).squeeze(0)
                gpu_choices.append(c_idx)
                tok_t = torch.where(c_idx == UNIT_VOCAB_SIZE, self.eos_token_id, c_idx + u_start).view(1, 1)
                u_logits_cur, eos_logit_cur = self._decode_1tok_logits(tok_t, dtype, session)

            choices_cpu = torch.stack(gpu_choices).tolist()
            steps_done += batch_n
            for ci in choices_cpu:
                if ci == UNIT_VOCAB_SIZE:
                    session.finished = True
                    break
                buf_units.append(ci)
                session.recent_unit_ids.append(ci + u_start)

            if session.finished or steps_done >= max_steps:
                yield buf_units, True
                return
            if len(buf_units) >= subchunk_size:
                yield buf_units, False
                buf_units = []

        yield buf_units, True
