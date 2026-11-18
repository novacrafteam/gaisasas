"""Shared helpers: system monitoring and chat-prompt construction (used by train / evaluate / chat UI / GUI)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from config import SYSTEM_PROMPTS


# ----------------------------------------------------------------------------- monitoring
def gpu_stats() -> dict:
    """GPU memory/temperature via nvidia-smi (works without torch). Empty dict if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        name, used, total, temp, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_name": name, "vram_used_mb": float(used), "vram_total_mb": float(total),
                "gpu_temp_c": float(temp), "gpu_util": float(util)}
    except Exception:  # noqa: BLE001
        return {}


def ram_stats() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"ram_used_gb": round((vm.total - vm.available) / 1e9, 2), "ram_total_gb": round(vm.total / 1e9, 2)}
    except Exception:  # noqa: BLE001
        return {}


def append_metric(path: str | Path, record: dict) -> None:
    record = {"time": time.time(), **record}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_metrics(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # last line may be half-written while training
    return out


# ----------------------------------------------------------------------------- prompts
def with_system(messages: list[dict], lang: str) -> list[dict]:
    if messages and messages[0]["role"] == "system":
        return list(messages)
    sp = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    return [{"role": "system", "content": sp}] + list(messages)


def build_prompt(tokenizer, messages: list[dict], lang: str = "en") -> str:
    """Prompt text ending right where the assistant answer starts.
    Qwen3.5 defaults to 'thinking'; GorphinAI is trained in NON-thinking mode (enable_thinking=False),
    so training and inference must both use this exact function."""
    return tokenizer.apply_chat_template(
        with_system(messages, lang), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_reply(model, tokenizer, messages: list[dict], lang: str = "en", max_new_tokens: int = 256,
                   temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.05) -> str:
    import torch

    prompt = build_prompt(tokenizer, messages, lang)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    do_sample = temperature > 0
    kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=20)
    else:
        kw.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**enc, **kw)
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def latest_checkpoint(ckpt_dir: str | Path) -> Optional[str]:
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    cks = [p for p in d.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    if not cks:
        return None
    return str(max(cks, key=lambda p: int(p.name.split("-")[-1])))
