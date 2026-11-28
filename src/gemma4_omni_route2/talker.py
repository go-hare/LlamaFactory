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


IGNORE_INDEX = -100
UNIT_VOCAB_SIZE = 6561
DUPLEX_SPECIAL_TOKENS = [f"<unused{i}>" for i in range(8)]


def ensure_gemma4_duplex_special_tokens(gemma_tokenizer) -> None:
    """Ensure <unused0>..<unused7> encode as single atomic token IDs 6..13."""
    test_ids = gemma_tokenizer.encode("<unused0>", add_special_tokens=False)
    if test_ids != [6]:
        gemma_tokenizer.add_tokens(DUPLEX_SPECIAL_TOKENS, special_tokens=True)


@dataclass
class TalkerStreamSession:
    """Per-connection incremental KV-cached streaming state for Gemma4SpeechGenerator."""

    kv_cache: DynamicCache = field(default_factory=DynamicCache)
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
        temperature: float = 0.0,
        top_k: int = 25,
    ) -> list[int]:
        """Incremental KV-cached unit generation (prefills ONLY new_embeds, decodes <=W units in ~4ms on A100)."""
        device = new_embeds.device
        dtype = new_embeds.dtype
        embed_fn = self.model.get_input_embeddings()

        if is_finished:
            sep_emb = embed_fn(torch.tensor([self.sep_token_id], dtype=torch.long, device=device)).to(dtype=dtype)
            new_embeds = torch.cat([new_embeds, sep_emb], dim=0)

        out = self.model(
            inputs_embeds=new_embeds.unsqueeze(0),
            past_key_values=session.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[0, -1, :].float()

        if not is_finished:
            max_steps = self.write_len
        else:
            est_cap = max(24, session.spoken_char_count * 7 - len(session.recent_unit_ids) + 18)
            max_steps = min(180, est_cap) if session.spoken_char_count > 0 else 180
        emitted_units: list[int] = []
        u_start = self.unit_offset
        u_end = self.unit_offset + UNIT_VOCAB_SIZE  # 158227

        for _ in range(max_steps):
            # Restrict strictly to speech units <0>..<6560> (plus <eos> once is_finished=True)
            allowed_logits = torch.full_like(logits, float("-inf"))
            allowed_logits[u_start:u_end] = logits[u_start:u_end]
            if is_finished:
                allowed_logits[self.eos_token_id] = logits[self.eos_token_id]

            # Guard ONLY against pathological >8-frame stuck loops (preserve normal 2~5 frame vowel nuclei!)
            if len(session.recent_unit_ids) >= 8 and len(set(session.recent_unit_ids[-8:])) == 1:
                stuck_id = session.recent_unit_ids[-1]
                allowed_logits[stuck_id] -= 8.0

            if temperature <= 1e-3:
                next_id = int(torch.argmax(allowed_logits).item())
            else:
                scaled = allowed_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, UNIT_VOCAB_SIZE + 1))
                    scaled[scaled < v[-1]] = float("-inf")
                probs = F.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            # Append chosen token embedding into KV cache so next Read chunk sees exact autoregressive history
            next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if next_id == self.eos_token_id:
                session.finished = True
                break

            unit_val = next_id - u_start
            if 0 <= unit_val < UNIT_VOCAB_SIZE:
                emitted_units.append(unit_val)
                session.recent_unit_ids.append(next_id)

            if len(emitted_units) < max_steps:
                step_out = self.model(
                    inputs_embeds=embed_fn(next_t).to(dtype=dtype),
                    past_key_values=session.kv_cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = step_out.logits[0, -1, :].float()
            else:
                # Feed the last unit of a non-final chunk into KV cache before returning
                _ = self.model(
                    inputs_embeds=embed_fn(next_t).to(dtype=dtype),
                    past_key_values=session.kv_cache,
                    use_cache=True,
                    return_dict=True,
                )

        return emitted_units

    @torch.inference_mode()
    def step_gemma_token(
        self,
        gemma_tokenizer,
        gemma_token_id: int,
        gemma_pred_hidden: torch.Tensor,
        session: TalkerStreamSession,
        is_turn_end: bool = False,
        temperature: float = 0.7,
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
