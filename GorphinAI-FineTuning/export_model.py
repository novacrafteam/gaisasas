"""Export: zip the LoRA adapter and optionally merge it into a full copy of the base model.

  python export_model.py --adapter outputs/adapter --zip                 # outputs/gorphinai_adapter.zip
  python export_model.py --adapter outputs/adapter --merge outputs/merged   # needs RAM ~ 2x model size
The original Hugging Face weights are only read, never modified. Merging writes NEW files.
Merge is done on the 16-bit model (not a 4-bit one) - merging into quantized weights is lossy/unsupported.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from config import TrainConfig, setup_logging

log = logging.getLogger("gorphinai")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--merge", metavar="OUT_DIR")
    args = ap.parse_args(argv)
    setup_logging()
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        log.error("No adapter_config.json in %s", adapter)
        return 2
    if not args.zip and not args.merge:
        ap.error("choose --zip and/or --merge OUT_DIR")
    if args.zip:
        target = shutil.make_archive(str(adapter.parent / "gorphinai_adapter"), "zip", root_dir=adapter)
        log.info("Adapter archive: %s", target)
    if args.merge:
        import torch
        from peft import PeftModel

        from model_loader import load_base_model, load_tokenizer

        cfg = TrainConfig()
        cfg.use_qlora = False
        cfg.precision = "fp16" if torch.cuda.is_available() else "fp32"
        base = load_base_model(cfg, for_training=False, device_map="cpu" if not torch.cuda.is_available() else None)
        merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
        out = Path(args.merge)
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out), safe_serialization=True)
        load_tokenizer(cfg.model_id).save_pretrained(str(out))
        log.info("Merged model saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
