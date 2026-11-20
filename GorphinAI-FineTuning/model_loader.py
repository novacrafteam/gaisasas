"""Loading of the ORIGINAL Qwen/Qwen3.5-2B weights (never modified) + optional LoRA adapter."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from config import MODEL_ID, TrainConfig

log = logging.getLogger("gorphinai")


def resolve_precision(precision: str):
    """Return (torch_dtype, use_fp16_amp, use_bf16_amp). GTX 1660 Ti / T4 have no bf16 -> fp16."""
    import torch

    if precision == "auto":
        if torch.cuda.is_available():
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    if precision == "bf16":
        return torch.bfloat16, False, True
    if precision == "fp16":
        return torch.float16, True, False
    return torch.float32, False, False


def check_environment() -> list[str]:
    """Return a list of human-readable warnings about the software stack."""
    warnings: list[str] = []
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        minor = int(transformers.__version__.split(".")[1])
        if (major, minor) < (4, 57):
            warnings.append(
                f"transformers {transformers.__version__} is too old for Qwen3.5 (model_type 'qwen3_5'). "
                "The model card requires the latest transformers (pip install -U transformers)."
            )
    except Exception as e:  # noqa: BLE001
        warnings.append(f"transformers not importable: {e}")
    try:
        import peft  # noqa: F401
    except Exception as e:  # noqa: BLE001
        warnings.append(f"peft not importable: {e}")
    try:
        import fla  # noqa: F401
    except Exception:  # noqa: BLE001
        warnings.append(
            "flash-linear-attention ('fla') is not installed: Qwen3.5 linear-attention layers will use a slow "
            "PyTorch fallback (training works but is much slower)."
        )
    return warnings


def load_tokenizer(model_id: str = MODEL_ID, revision: Optional[str] = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(cfg: TrainConfig, for_training: bool = True, device_map=None):
    """Load base weights. Tries the causal-LM auto class first, then the image-text class
    (the hub checkpoint's architecture is Qwen3_5ForConditionalGeneration)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    dtype, _, _ = resolve_precision(cfg.precision)
    kwargs = dict(revision=cfg.revision, trust_remote_code=cfg.trust_remote_code, low_cpu_mem_usage=True)

    if cfg.use_qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs a CUDA GPU. Use --no-qlora or run on a GPU machine.")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"QLoRA requested but bitsandbytes is unavailable: {e}") from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
        )
        kwargs["device_map"] = device_map or {"": 0}
    else:
        kwargs["dtype"] = dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = device_map or {"": 0}

    last_err: Exception | None = None
    classes = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    if hasattr(transformers, "AutoModelForImageTextToText"):
        classes.append(("AutoModelForImageTextToText", transformers.AutoModelForImageTextToText))
    for name, cls in classes:
        try:
            log.info("Loading %s with %s ...", cfg.model_id, name)
            try:
                model = cls.from_pretrained(cfg.model_id, **kwargs)
            except TypeError:
                # older transformers use torch_dtype instead of dtype
                if "dtype" in kwargs:
                    kwargs["torch_dtype"] = kwargs.pop("dtype")
                model = cls.from_pretrained(cfg.model_id, **kwargs)
            model.config.use_cache = not for_training
            return model
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("%s failed: %s", name, e)
    raise RuntimeError(
        f"Could not load '{cfg.model_id}'. Check the model ID, internet access and that transformers is recent "
        f"enough for Qwen3.5. Last error: {last_err}"
    )


def attach_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_regex,  # a str is treated by PEFT as a regex (full-match)
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("LoRA attached to 0 parameters: target regex matched nothing for this architecture.")
    peft_model.print_trainable_parameters()
    return peft_model


def load_with_adapter(adapter_dir: str | Path, cfg: Optional[TrainConfig] = None) -> Tuple[object, object]:
    """Load base model + trained adapter for inference. Returns (peft_model, tokenizer)."""
    from peft import PeftModel

    cfg = cfg or TrainConfig()
    tok = load_tokenizer(cfg.model_id, cfg.revision)
    base = load_base_model(cfg, for_training=False)
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tok


def estimate_memory_gb(cfg: TrainConfig, n_params_b: float = 4.5) -> dict:
    """ROUGH VRAM estimate (GB). Not a guarantee: real usage depends on kernels and sequence content."""
    weights = n_params_b * (0.55 if cfg.use_qlora else 2.0)   # nf4 ~0.5 B/param (+ embeddings in fp16)
    if cfg.use_qlora:
        weights += 1.0                                        # 248k x 2048 embeddings stay 16-bit
    lora = 0.15 * cfg.lora_r / 16 * 4                         # fp32 params + Adam states, small
    act = 0.0006 * cfg.max_seq_length * cfg.per_device_train_batch_size * (1 if cfg.gradient_checkpointing else 12)
    logits = cfg.max_seq_length * cfg.per_device_train_batch_size * 248320 * 4 * 2 / 1e9  # fp32 logits + grad
    total = weights + lora + act + logits + 0.8
    return {"weights": round(weights, 2), "lora+optim": round(lora, 2), "activations": round(act, 2),
            "logits": round(logits, 2), "total_rough": round(total, 2)}
