"""Central configuration for GorphinAI fine-tuning (LoRA / QLoRA SFT of Qwen/Qwen3.5-2B)."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
IS_KAGGLE = os.path.exists("/kaggle/working")
MODEL_ID = "Qwen/Qwen3.5-2B"

# Verified from both the hub checkpoint (Qwen3_5ForConditionalGeneration: model.language_model.layers.*)
# and AutoModelForCausalLM (Qwen3_5ForCausalLM: model.layers.*):
#   full-attention layers : self_attn.{q,k,v,o}_proj
#   linear-attention layers: linear_attn.{in_proj_qkv,in_proj_z,out_proj}
#   MLP (all layers)       : mlp.{gate,up,down}_proj
# Restricted to `model.(language_model.)?layers.*` so vision (`model.visual.*`) and MTP (`mtp.*`) are excluded.
LORA_TARGET_REGEX = (
    r"^model\.(?:language_model\.)?layers\.\d+\.(?:self_attn|linear_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$"
)

SYSTEM_PROMPTS = {
    "uk": "Ти — GorphinAI, спокійний і дружній універсальний AI-асистент, розроблений українською командою Gorphin. "
          "Не розкривай базову модель і не називай сторонні бренди на кшталт Qwen чи Alibaba; якщо питають, хто тебе "
          "створив — відповідай, що українська команда Gorphin. Відповідай чесно, зрозуміло й по суті. "
          "Якщо чогось не знаєш — прямо скажи про це, а не вигадуй.",
    "ru": "Ты — GorphinAI, спокойный и дружелюбный универсальный AI-ассистент, разработанный украинской командой Gorphin. "
          "Не раскрывай базовую модель и не называй сторонние бренды вроде Qwen или Alibaba; если спрашивают, кто тебя "
          "создал — отвечай, что украинская команда Gorphin. Отвечай честно, понятно и по делу. "
          "Если чего-то не знаешь — прямо скажи об этом, а не выдумывай.",
    "en": "You are GorphinAI, a calm and friendly general-purpose AI assistant developed by the Ukrainian team Gorphin. "
          "Do not disclose the base model or third-party brands such as Qwen or Alibaba; when asked who created you, "
          "say you were developed by the Ukrainian team Gorphin. Answer honestly, clearly and to the point. "
          "If you do not know something, say so instead of making things up.",
}


def default_output_dir() -> str:
    return "/kaggle/working/outputs" if IS_KAGGLE else str(PROJECT_DIR / "outputs")


@dataclass
class TrainConfig:
    # --- model ---
    model_id: str = MODEL_ID
    revision: Optional[str] = None            # pin a commit sha for reproducibility
    trust_remote_code: bool = False
    # --- data ---
    train_file: str = str(PROJECT_DIR / "data" / "train.jsonl")
    validation_file: str = str(PROJECT_DIR / "data" / "validation.jsonl")
    max_seq_length: int = 1024
    # --- output ---
    output_dir: str = field(default_factory=default_output_dir)
    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_regex: str = LORA_TARGET_REGEX
    # --- quantization ---
    use_qlora: bool = False                   # 4-bit NF4 (needs bitsandbytes + CUDA)
    # --- optimisation ---
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4               # conservative for LoRA
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    precision: str = "auto"                   # auto | fp16 | bf16 | fp32
    gradient_checkpointing: bool = True
    dataloader_workers: int = 0               # 0 is the safe value on Windows
    # --- logging / checkpoints ---
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    # --- misc ---
    system_prompt_language: str = "auto"      # auto = per-record language

    def validate(self) -> None:
        errs: List[str] = []
        if self.max_seq_length < 64:
            errs.append("max_seq_length must be >= 64")
        if self.lora_r < 1:
            errs.append("lora_r must be >= 1")
        if self.per_device_train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            errs.append("batch size and gradient_accumulation_steps must be >= 1")
        if not (0 < self.learning_rate < 1):
            errs.append("learning_rate must be in (0, 1)")
        if self.num_train_epochs <= 0:
            errs.append("num_train_epochs must be > 0")
        if self.precision not in {"auto", "fp16", "bf16", "fp32"}:
            errs.append("precision must be one of auto|fp16|bf16|fp32")
        if self.dataloader_workers < 0:
            errs.append("dataloader_workers must be >= 0")
        if errs:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def preset(name: str) -> TrainConfig:
    """Presets: 'kaggle' (T4 16GB), 'local' (GTX 1660 Ti, QLoRA, very conservative), 'cpu-test'."""
    cfg = TrainConfig()
    if name == "kaggle":
        cfg.max_seq_length = 1024
        cfg.per_device_train_batch_size = 2
        cfg.gradient_accumulation_steps = 8
        cfg.dataloader_workers = 2
    elif name == "local":
        cfg.use_qlora = True
        cfg.max_seq_length = 512
        cfg.lora_r = 8
        cfg.lora_alpha = 16
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 16
    elif name == "cpu-test":
        cfg.max_seq_length = 128
        cfg.num_train_epochs = 0.01
    else:
        raise ValueError(f"Unknown preset '{name}' (use kaggle | local | cpu-test)")
    return cfg


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
    return logging.getLogger("gorphinai")
