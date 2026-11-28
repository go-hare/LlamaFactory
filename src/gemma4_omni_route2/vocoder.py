"""Self-contained CosyVoice2 25Hz Speech Tokenizer & 24kHz Streaming Flow-HiFT Vocoder.

Zero dependency on WeTextProcessing / ttsfrd / pynini / Lightning / CosyVoice-BlankEN.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import snapshot_download
from omegaconf import DictConfig


_VENDOR_DIR = str(Path(__file__).resolve().parent / "_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from cosyvoice.flow.decoder import ConditionalDecoder  # noqa: E402
from cosyvoice.flow.flow import CausalMaskedDiffWithXvec  # noqa: E402
from cosyvoice.flow.flow_matching import CausalConditionalCFM  # noqa: E402
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor  # noqa: E402
from cosyvoice.hifigan.generator import HiFTGenerator  # noqa: E402
from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder  # noqa: E402


DEFAULT_PROMPT_ZH_WAV = str(Path(__file__).resolve().parent / "assets" / "prompt_zh.wav")


def download_cosy2_assets(repo_id: str = "ICTNLP/cosy2_decoder", local_dir: Optional[str] = None) -> str:
    """Download only the lightweight (~1.05 GB) tokenizer + flow + hift weights needed for Route 2."""
    kwargs = {
        "repo_id": repo_id,
        "allow_patterns": [
            "campplus.onnx",
            "speech_tokenizer_v2.onnx",
            "flow.pt",
            "hift.pt",
        ],
    }
    if local_dir is not None:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def load_mono_audio(path_or_array, target_sr: int = 16000) -> torch.Tensor:
    """Load mono audio as [1, T] float32 tensor resampled to target_sr."""
    if isinstance(path_or_array, (str, Path)):
        wav_np, sr = sf.read(str(path_or_array), dtype="float32", always_2d=False)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
    elif isinstance(path_or_array, np.ndarray):
        arr = path_or_array.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        wav = torch.from_numpy(arr).unsqueeze(0)
        sr = target_sr
    elif isinstance(path_or_array, torch.Tensor):
        wav = path_or_array.detach().float().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        sr = target_sr
    else:
        raise TypeError(f"Unsupported audio input: {type(path_or_array)}")

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav


def extract_cosy2_mel_24k(speech_24k: torch.Tensor) -> torch.Tensor:
    """Exact Matcha/CosyVoice2 24kHz 80-bin mel spectrogram (n_fft=1920, hop=480, win=1920, fmax=8000, center=False)."""
    device = speech_24k.device
    y = speech_24k.float()
    pad = (1920 - 480) // 2  # 720
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    hann = torch.hann_window(1920, device=device)
    spec = torch.view_as_real(
        torch.stft(y, n_fft=1920, hop_length=480, win_length=1920, window=hann, center=False, return_complex=True)
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=1920 // 2 + 1,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        sample_rate=24000,
        norm="slaney",
        mel_scale="slaney",
    ).to(device)
    mel = torch.matmul(fb.T, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel  # [B, 80, T_mel]


def _fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: np.ndarray) -> torch.Tensor:
    device = fade_in_mel.device
    fade_in_cpu = fade_in_mel.cpu()
    fade_out_cpu = fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] // 2)
    win_t = torch.from_numpy(window).to(fade_in_cpu.dtype)
    fade_in_cpu[..., :mel_overlap_len] = (
        fade_in_cpu[..., :mel_overlap_len] * win_t[:mel_overlap_len]
        + fade_out_cpu[..., -mel_overlap_len:] * win_t[mel_overlap_len:]
    )
    return fade_in_cpu.to(device)


def extract_whisper_log_mel_128(wav_16k: torch.Tensor) -> torch.Tensor:
    """Bit-exact pure-PyTorch Whisper 128-bin log-mel spectrogram (zero openai-whisper dependency)."""
    pad_samples = 16000 * 30
    if wav_16k.shape[-1] < pad_samples:
        wav_p = F.pad(wav_16k, (0, pad_samples - wav_16k.shape[-1]))
    else:
        wav_p = wav_16k[..., :pad_samples]
    window = torch.hann_window(400, device=wav_p.device)
    stft = torch.stft(wav_p.float(), n_fft=400, hop_length=160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torchaudio.functional.melscale_fbanks(
        n_freqs=201, f_min=0.0, f_max=8000.0, n_mels=128, sample_rate=16000, norm="slaney", mel_scale="slaney"
    ).T.to(wav_p.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


class SpeechTokenizerV2:
    """Extracts exact 25Hz discrete FSQ speech units (0..6560) using CosyVoice2 speech_tokenizer_v2.onnx."""

    def __init__(self, onnx_path: str, prefer_cuda: bool = True):
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 2
        avail = onnxruntime.get_available_providers()
        if prefer_cuda and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(onnx_path, sess_options=opt, providers=providers)
        self.in0 = self.session.get_inputs()[0].name
        self.in1 = self.session.get_inputs()[1].name

    def extract_units(self, audio_16k) -> list[int]:
        """Extract 25Hz discrete speech tokens (0..6560) from up to 30s of 16kHz mono audio."""
        wav = load_mono_audio(audio_16k, target_sr=16000)
        if wav.shape[1] > 16000 * 30:
            wav = wav[:, : 16000 * 30]
        if wav.shape[1] < 400:
            return []
        valid_frames = wav.shape[1] // 160
        feat = extract_whisper_log_mel_128(wav)[..., :valid_frames]  # [1, 128, T_mel]
        out = self.session.run(
            None,
            {
                self.in0: feat.detach().cpu().numpy(),
                self.in1: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0]
        return out.flatten().astype(int).tolist()


def _build_cosy2_flow_and_hift() -> tuple[CausalMaskedDiffWithXvec, HiFTGenerator]:
    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        input_size=512,
        use_cnn_module=False,
        macaron_style=False,
    )
    estimator = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    )
    cfm_params = DictConfig(
        {
            "sigma_min": 1e-6,
            "solver": "euler",
            "t_scheduler": "cosine",
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": "l1",
        }
    )
    decoder = CausalConditionalCFM(
        in_channels=240,
        cfm_params=cfm_params,
        n_spks=1,
        spk_emb_dim=80,
        estimator=estimator,
    )
    flow = CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=6561,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=decoder,
    )
    f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    hift = HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=f0_predictor,
    )
    return flow, hift


class Cosy2StreamingVocoder:
    """Streaming 25Hz Speech-Token -> 24kHz Waveform Vocoder with 0ms Barge-In Session Flush."""

    sample_rate: int = 24000

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        use_zero_shot_prompt: bool = True,
    ):
        if model_dir is None:
            model_dir = download_cosy2_assets()
        self.model_dir = model_dir
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1
        self.campplus = onnxruntime.InferenceSession(
            os.path.join(model_dir, "campplus.onnx"),
            sess_options=opt,
            providers=["CPUExecutionProvider"],
        )
        # Run one-time prompt unit extraction strictly on CPU so onnxruntime never hijacks CUDA VRAM arena!
        speech_tok_cpu = SpeechTokenizerV2(
            os.path.join(model_dir, "speech_tokenizer_v2.onnx"),
            prefer_cuda=False,
        )

        self.flow, self.hift = _build_cosy2_flow_and_hift()
        self.flow.load_state_dict(
            torch.load(os.path.join(model_dir, "flow.pt"), map_location=self.device, weights_only=True),
            strict=True,
        )
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False

        raw_hift = torch.load(os.path.join(model_dir, "hift.pt"), map_location=self.device, weights_only=True)
        hift_sd = {k.replace("generator.", ""): v for k, v in raw_hift.items()}
        self.hift.load_state_dict(hift_sd, strict=True)
        self.hift.to(self.device).eval()

        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = (
            2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        )
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.lock = threading.Lock()
        self.hift_cache_dict: dict[str, Optional[dict]] = {}

        ref_wav = prompt_wav_path or DEFAULT_PROMPT_ZH_WAV
        self.default_speaker_ctx = self.prepare_speaker_context(
            ref_wav, use_zero_shot_prompt=use_zero_shot_prompt, speech_tokenizer=speech_tok_cpu
        )
        del speech_tok_cpu

    def extract_spk_embedding(self, speech_16k: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.campplus.run(
            None,
            {self.campplus.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
        )[0].flatten()
        return torch.from_numpy(emb).unsqueeze(0).float().to(self.device)

    def prepare_speaker_context(
        self,
        prompt_wav_path: str,
        use_zero_shot_prompt: bool = True,
        speech_tokenizer: Optional[SpeechTokenizerV2] = None,
    ) -> dict[str, torch.Tensor]:
        speech_16k = load_mono_audio(prompt_wav_path, target_sr=16000)
        # Keep prompt crisp (~1.5s) so streaming flow latency stays <18ms per chunk on CUDA
        if speech_16k.shape[1] > int(16000 * 1.5):
            speech_16k = speech_16k[:, : int(16000 * 1.5)]
        embedding = self.extract_spk_embedding(speech_16k)
        if not use_zero_shot_prompt:
            return {
                "embedding": embedding,
                "prompt_token": torch.zeros(1, 0, dtype=torch.int32, device=self.device),
                "prompt_feat": torch.zeros(1, 0, 80, dtype=torch.float32, device=self.device),
            }
        tok_inst = speech_tokenizer or SpeechTokenizerV2(
            os.path.join(self.model_dir, "speech_tokenizer_v2.onnx"), prefer_cuda=False
        )
        units = tok_inst.extract_units(speech_16k)
        speech_24k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(speech_16k)
        mel = extract_cosy2_mel_24k(speech_24k).transpose(1, 2).to(self.device)  # [1, T_mel, 80]
        tok_len = min(mel.shape[1] // 2, len(units))
        prompt_feat = mel[:, : 2 * tok_len, :].contiguous()
        prompt_token = torch.tensor([units[:tok_len]], dtype=torch.int32, device=self.device)
        return {
            "embedding": embedding,
            "prompt_token": prompt_token,
            "prompt_feat": prompt_feat,
        }

    def init_session(self, speaker_ctx: Optional[dict[str, torch.Tensor]] = None) -> dict:
        ctx = speaker_ctx or self.default_speaker_ctx
        sid = str(uuid.uuid1())
        with self.lock:
            self.hift_cache_dict[sid] = None
        return {
            "uuid": sid,
            "embedding": ctx["embedding"],
            "prompt_token": ctx["prompt_token"],
            "prompt_feat": ctx["prompt_feat"],
            "token_offset": 0,
            "generated_tokens": [],
        }

    def abort_session(self, session: Optional[dict]) -> None:
        """Instant 0ms flush on <unused6> barge-in."""
        if not session:
            return
        sid = session.get("uuid")
        with self.lock:
            if sid in self.hift_cache_dict:
                del self.hift_cache_dict[sid]
        session["generated_tokens"] = []
        session["token_offset"] = 0

    @torch.inference_mode()
    def _token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        sid: str,
        token_offset: int,
        finalize: bool = False,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=self.device),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=self.device),
                embedding=embedding,
                finalize=finalize,
            )
        tts_mel = tts_mel.float()
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio :]
        if tts_mel.shape[2] == 0:
            return torch.zeros(1, 0, dtype=torch.float32, device=self.device)

        cache_entry = self.hift_cache_dict.get(sid)
        if cache_entry is not None:
            hift_cache_mel = cache_entry["mel"]
            hift_cache_source = cache_entry["source"]
            tts_mel = torch.cat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0, device=self.device)

        if not finalize:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if cache_entry is not None:
                tts_speech = _fade_in_out(tts_speech, cache_entry["speech"], self.speech_window)
            self.hift_cache_dict[sid] = {
                "mel": tts_mel[:, :, -self.mel_cache_len :].detach(),
                "source": tts_source[:, :, -self.source_cache_len :].detach(),
                "speech": tts_speech[:, -self.source_cache_len :].detach(),
            }
            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if cache_entry is not None:
                tts_speech = _fade_in_out(tts_speech, cache_entry["speech"], self.speech_window)
            with self.lock:
                self.hift_cache_dict.pop(sid, None)
        return tts_speech

    @torch.inference_mode()
    def push_units(
        self,
        new_units: list[int],
        session: dict,
        finalize: bool = False,
        min_chunk_units: int = 6,
    ) -> np.ndarray:
        """Push newly generated 25Hz units (0..6560) and return ready 24kHz float32 PCM samples."""
        if new_units:
            valid = [int(u) for u in new_units if 0 <= int(u) < 6561]
            session["generated_tokens"].extend(valid)

        total_len = len(session["generated_tokens"])
        offset = session["token_offset"]
        avail = total_len - offset

        if total_len == 0:
            if finalize:
                self.abort_session(session)
            return np.zeros(0, dtype=np.float32)

        if not finalize:
            # Need at least min_chunk_units + pre_lookahead_len so mel > mel_cache_len (8 frames = 4 tokens)
            if avail < max(min_chunk_units, 5) + self.flow.pre_lookahead_len:
                return np.zeros(0, dtype=np.float32)
        else:
            if avail <= 0:
                self.abort_session(session)
                return np.zeros(0, dtype=np.float32)

        tok_tensor = torch.tensor([session["generated_tokens"]], dtype=torch.int32, device=self.device)
        wav = self._token2wav(
            token=tok_tensor,
            prompt_token=session["prompt_token"],
            prompt_feat=session["prompt_feat"],
            embedding=session["embedding"],
            sid=session["uuid"],
            token_offset=offset,
            finalize=finalize,
        )
        if not finalize:
            session["token_offset"] = total_len - self.flow.pre_lookahead_len
        else:
            session["token_offset"] = total_len
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def synthesize_full(self, units: list[int], speaker_ctx: Optional[dict[str, torch.Tensor]] = None) -> np.ndarray:
        """One-shot synthesis of a full 25Hz unit list into 24kHz float32 PCM."""
        sess = self.init_session(speaker_ctx=speaker_ctx)
        return self.push_units(units, sess, finalize=True)
