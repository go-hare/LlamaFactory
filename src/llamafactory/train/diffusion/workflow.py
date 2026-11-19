"""Diffusion SFT entry. Reads prompt/response JSONL already wrapped for Gemma 4."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .loss import SelfConditioner, _get_conditioner, diffusion_batch_step, prepare_conversion_training


logger = get_logger(__name__)


class DiffusionJsonl(Dataset):
    def __init__(self, path: str, tokenizer, prompt_len: int, response_len: int, *, prompt_only: bool = False):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.response_len = response_len
        self.prompt_only = prompt_only
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw_line = handle.readline().decode("utf-8")
        row = json.loads(raw_line)
        prompt = self.tokenizer.encode(row["prompt"], add_special_tokens=False)[: self.prompt_len]
        response = (
            []
            if self.prompt_only
            else self.tokenizer.encode(row.get("response") or "", add_special_tokens=False)[: self.response_len]
        )
        if not prompt:
            prompt = [self.tokenizer.bos_token_id or 2]
        if not response and not self.prompt_only:
            response = [self.tokenizer.eos_token_id or 1]
        answer = str(row.get("answer") or "")
        return {
            "prompt": torch.tensor(prompt, dtype=torch.long),
            "response": torch.tensor(response, dtype=torch.long),
            "answer": answer,
        }


class DiffusionTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        canvas_size: int,
        num_canvases: int,
        pad_id: int,
        eos_id: int,
        finetuning_args=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.canvas_size = canvas_size
        self.num_canvases = num_canvases
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.finetuning_args = finetuning_args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del return_outputs, num_items_in_batch
        prompts, responses = inputs["prompt"], inputs["response"]
        loss, parts = diffusion_batch_step(
            model,
            prompts,
            responses,
            canvas_size=self.canvas_size,
            num_canvases=self.num_canvases,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def get_decay_parameter_names(self, model) -> list[str]:
        # Hugging Face drops every name containing "norm". The conditioner pre-norm
        # is a real weight and must decay with the other matrices.
        names = super().get_decay_parameter_names(model)
        extra = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "self_conditioner" in name and name not in names and "bias" not in name
        ]
        return names + extra

    def create_optimizer(self, model=None):
        # The parent is a plain Seq2SeqTrainer, so the Factory LoRA optimizer
        # would otherwise never be built and LoRA+ would be ignored.
        if self.optimizer is None:
            target_model = model if model is not None else self.model
            use_master = bool(getattr(self.finetuning_args, "diffusion_master_weights", True))
            has_bf16_trainable = any(
                p.requires_grad and p.dtype == torch.bfloat16 for p in target_model.parameters()
            )
            if use_master and has_bf16_trainable and getattr(self.finetuning_args, "finetuning_type", "lora") == "full":
                from .precision import MasterAdamW

                decay_names = set(self.get_decay_parameter_names(target_model))
                decay_params = [
                    p for n, p in target_model.named_parameters() if p.requires_grad and n in decay_names
                ]
                nodecay_params = [
                    p for n, p in target_model.named_parameters() if p.requires_grad and n not in decay_names
                ]
                groups = [
                    {"params": decay_params, "weight_decay": self.args.weight_decay},
                    {"params": nodecay_params, "weight_decay": 0.0},
                ]
                self.optimizer = MasterAdamW(
                    [g for g in groups if g["params"]],
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                )
            else:
                self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(model)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: str | None = None, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        target_dir = Path(output_dir or self.args.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        conditioner = _get_conditioner(self.model)
        raw_cond = getattr(conditioner, "modules_to_save", None)
        if raw_cond is not None and "default" in raw_cond:
            conditioner = raw_cond["default"]
        from safetensors.torch import save_file

        save_file(
            {k: v.detach().cpu().contiguous() for k, v in conditioner.state_dict().items()},
            str(target_dir / "self_conditioner.safetensors"),
        )


def _load_conditioner_weights(conditioner: SelfConditioner, *dirs: str | None) -> bool:
    from safetensors import safe_open
    from safetensors.torch import load_file

    for d in dirs:
        if not d:
            continue
        p = Path(d)
        if not p.is_dir():
            continue
        sc_file = p / "self_conditioner.safetensors"
        if sc_file.is_file():
            conditioner.load_state_dict(load_file(str(sc_file)), strict=True)
            logger.info_rank0(f"Loaded self_conditioner from {sc_file}")
            return True
        for candidate in sorted(p.glob("*.safetensors")):
            with safe_open(str(candidate), framework="pt", device="cpu") as f:
                keys = [k for k in f.keys() if "self_conditioner." in k]
                if keys:
                    sub = {}
                    for k in keys:
                        short = k.split("self_conditioner.", 1)[1]
                        if short.startswith("modules_to_save.default."):
                            short = short[len("modules_to_save.default.") :]
                        if not short.startswith("original_module."):
                            sub[short] = f.get_tensor(k)
                    if sub:
                        conditioner.load_state_dict(sub, strict=True)
                        logger.info_rank0(f"Loaded self_conditioner from {candidate}")
                        return True
    return False


def _dataset_name(value, default: str = "train") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else default
    return str(value).split(",")[0]


def _split_path(data_args, name: str) -> str:
    root = Path(data_args.dataset_dir)
    path = root / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Diffusion JSONL not found: {path}")
    return str(path)


def run_diffusion(model_args, data_args, training_args, finetuning_args, generating_args=None, callbacks=None):
    """Section 4 conversion training. ``dataset`` is the split name, default train."""
    del generating_args, callbacks
    from ...model import load_model, load_tokenizer

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # Register before the optimizer is built, or its gradient is never applied.
    # Peft only writes LoRA and modules_to_save, so the conditioner has to be one
    # of those or the saved adapter cannot be resumed.
    text_config = getattr(model.config, "text_config", model.config)
    hidden = int(getattr(text_config, "hidden_size", 0) or model.config.hidden_size)
    intermediate = int(getattr(text_config, "intermediate_size", 0) or hidden * 4)
    param = next(model.parameters())
    conditioner = SelfConditioner(hidden, intermediate).to(device=param.device, dtype=param.dtype)
    _load_conditioner_weights(
        conditioner,
        getattr(training_args, "resume_from_checkpoint", None),
        getattr(model_args, "model_name_or_path", None),
    )
    # PreTrainedModel now has add_adapter; that is not a PeftModel. Full
    # conversion just registers the module. LoRA still wraps it so peft saves it.
    is_peft = getattr(model, "peft_config", None) is not None
    if is_peft:
        try:
            from peft.utils.other import ModulesToSaveWrapper
        except ImportError:
            from peft.tuners.tuners_utils import ModulesToSaveWrapper

        model.base_model.model.self_conditioner = ModulesToSaveWrapper(conditioner, adapter_name="default")
    else:
        model.self_conditioner = conditioner
        if training_args.do_train:
            prepare_conversion_training(model)
    prompt_len = int(getattr(data_args, "cutoff_len", 1024))
    # Section 4: C=256. K=18 covers ~p99 response length (~4608 tokens).
    canvas_size, num_canvases = 256, 18
    train_name = _dataset_name(data_args.dataset, "train")
    train_data = DiffusionJsonl(_split_path(data_args, train_name), tokenizer, prompt_len, canvas_size * num_canvases)
    eval_data = None
    if training_args.do_eval and data_args.eval_dataset:
        eval_data = DiffusionJsonl(
            _split_path(data_args, _dataset_name(data_args.eval_dataset, "valid")),
            tokenizer,
            prompt_len,
            canvas_size * num_canvases,
        )
    logger.info_rank0(f"Diffusion rows: train={len(train_data)} eval={0 if eval_data is None else len(eval_data)}")
    trainer = DiffusionTrainer(
        model=model,
        args=training_args if isinstance(training_args, Seq2SeqTrainingArguments) else training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=lambda features: {
            "prompt": [row["prompt"] for row in features],
            "response": [row["response"] for row in features],
            "answer": [row["answer"] for row in features],
        },
        processing_class=tokenizer,
        canvas_size=canvas_size,
        num_canvases=num_canvases,
        pad_id=tokenizer.pad_token_id or 0,
        eos_id=tokenizer.eos_token_id or 1,
        finetuning_args=finetuning_args,
    )
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        _run_sd_rl(model, tokenizer, data_args, training_args, finetuning_args, trainer)


def _run_sd_rl(model, tokenizer, data_args, training_args, finetuning_args, trainer) -> None:
    """Continue from the SFT weights with the sampler-distillation stage.

    ``rl_steps`` of 0 leaves the run as SFT only. Rows come from ``rl.jsonl``
    when that file exists, otherwise from the SFT split. A row may omit
    ``response``; ``answer`` is the string a correct canvas must contain.
    """
    steps = int(getattr(finetuning_args, "rl_steps", 0) or 0)
    if steps <= 0:
        return
    from .rl import sd_rl_step

    root = Path(data_args.dataset_dir)
    rl_path = root / "rl.jsonl"
    source = rl_path if rl_path.exists() else Path(_split_path(data_args, _dataset_name(data_args.dataset, "train")))
    prompt_len = int(getattr(data_args, "cutoff_len", 1024))
    rows = DiffusionJsonl(str(source), tokenizer, prompt_len, 0, prompt_only=True)
    if len(rows) == 0:
        raise RuntimeError(f"SD-RL found no prompts in {source}.")
    logger.info_rank0(f"SD-RL steps={steps} prompts={len(rows)} from {source.name}")
    optimizer = trainer.create_optimizer()
    model.train()
    for step in range(steps):
        row = rows[step % len(rows)]
        loss, parts = sd_rl_step(
            model,
            row["prompt"].to(next(model.parameters()).device),
            answer=row["answer"],
            tokenizer=tokenizer,
            canvas_size=trainer.canvas_size,
            pad_id=tokenizer.pad_token_id or 0,
            teacher_steps=int(getattr(finetuning_args, "rl_teacher_steps", 8)),
            student_steps=int(getattr(finetuning_args, "rl_student_steps", 2)),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            logger.info_rank0(
                f"SD-RL {step} loss {float(loss.detach()):.4f} reward {parts['reward']:.0f} entropy {parts['entropy']:.3f}"
            )
    trainer.save_model()
