"""Convert a trained GorphinAI LoRA adapter into a .gguf file.

Examples:
  # 1) Standalone GGUF (merges adapter with Qwen/Qwen3.5-2B, then converts via llama.cpp):
  python convert_to_gguf.py --adapter ../gorphinai_adapter --out gorphinai.gguf

  # 2) LoRA-only GGUF (fast, small file; requires base GGUF at runtime):
  python convert_to_gguf.py --adapter ../gorphinai_adapter --out gorphinai-lora.gguf --lora-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import TrainConfig, setup_logging
from export_model import convert_to_gguf, resolve_llama_cpp

log = logging.getLogger("gorphinai")


def default_adapter_path() -> str:
    for candidate in ("../gorphinai_adapter", "gorphinai_adapter", "outputs/adapter"):
        if (Path(candidate) / "adapter_config.json").exists():
            return candidate
    return "outputs/adapter"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=default_adapter_path(), help="path to adapter directory (with adapter_config.json)")
    ap.add_argument("--out", help="output .gguf file path (default: outputs/gorphinai.gguf or gorphinai-lora.gguf)")
    ap.add_argument("--lora-only", action="store_true", help="convert only the LoRA adapter without merging base weights")
    ap.add_argument("--merged-dir", default="outputs/merged_for_gguf", help="where to store temporary/merged HF weights")
    ap.add_argument("--outtype", default="f16", help="GGUF tensor type: f16, bf16, q8_0, auto")
    ap.add_argument("--llama-cpp-dir", help="existing llama.cpp directory (optional)")
    ap.add_argument("--no-fetch", action="store_true", help="do not auto-clone llama.cpp if missing")
    args = ap.parse_args(argv)
    setup_logging()

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        log.error("No adapter_config.json found in %s", adapter)
        return 2

    out_gguf = Path(args.out) if args.out else Path("outputs") / ("gorphinai-lora.gguf" if args.lora_only else "gorphinai.gguf")

    try:
        llama_dir = resolve_llama_cpp(args.llama_cpp_dir, fetch=not args.no_fetch)
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        return 2

    if args.lora_only:
        try:
            convert_to_gguf(adapter, out_gguf, args.outtype, is_lora=True, llama_cpp_dir=llama_dir)
        except Exception as e:  # noqa: BLE001
            log.error("LoRA GGUF conversion failed: %s", e)
            return 1
        return 0

    import torch
    from peft import PeftModel

    from model_loader import load_base_model, load_tokenizer

    cfg = TrainConfig()
    cfg.use_qlora = False
    cfg.precision = "fp16" if torch.cuda.is_available() else "fp32"
    log.info("Merging adapter %s with %s (this requires ~5-10 GB RAM) ...", adapter, cfg.model_id)
    base = load_base_model(cfg, for_training=False, device_map="cpu" if not torch.cuda.is_available() else None)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    load_tokenizer(cfg.model_id).save_pretrained(str(merged_dir))
    log.info("Merged HF model saved to %s", merged_dir)

    try:
        convert_to_gguf(merged_dir, out_gguf, args.outtype, is_lora=False, llama_cpp_dir=llama_dir)
    except Exception as e:  # noqa: BLE001
        log.error("GGUF conversion failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
