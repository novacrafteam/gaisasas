"""Export: zip the LoRA adapter, save PyTorch (.pt), merge into base weights, and convert to GGUF (.gguf).

  python export_model.py --adapter outputs/adapter --zip --pt
  python export_model.py --adapter outputs/adapter --merge outputs/merged --pt
  python export_model.py --adapter outputs/adapter --merge outputs/merged --gguf outputs/gorphinai.gguf --fetch-llama-cpp

The original Hugging Face weights are only read, never modified. Merging writes NEW files.
Merge is done on the 16-bit model (not a 4-bit one) - merging into quantized weights is lossy/unsupported.
GGUF conversion uses llama.cpp (convert_hf_to_gguf.py for merged models, convert_lora_to_gguf.py for adapters).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import TrainConfig, setup_logging

log = logging.getLogger("gorphinai")


def save_adapter_pt(adapter: Path) -> Path:
    """Save adapter weights as a PyTorch .pt state_dict next to adapter_model.safetensors."""
    import torch

    out_pt = adapter / "adapter_model.pt"
    st_file = adapter / "adapter_model.safetensors"
    if st_file.exists():
        from safetensors.torch import load_file

        state = load_file(str(st_file))
    else:
        bin_file = adapter / "adapter_model.bin"
        if not bin_file.exists():
            raise FileNotFoundError(f"Neither adapter_model.safetensors nor adapter_model.bin found in {adapter}")
        state = torch.load(str(bin_file), map_location="cpu", weights_only=True)
    torch.save(state, str(out_pt))
    log.info("Adapter PyTorch weights saved to %s", out_pt)
    return out_pt


def resolve_llama_cpp(explicit_dir: str | None, fetch: bool) -> Path:
    candidates = [p for p in [explicit_dir, os.environ.get("LLAMA_CPP_DIR"), "/kaggle/working/llama.cpp", "llama.cpp"] if p]
    for c in candidates:
        p = Path(c)
        if (p / "convert_hf_to_gguf.py").exists():
            return p
    if not fetch:
        raise FileNotFoundError(
            "llama.cpp not found (needed for .gguf conversion). Pass --fetch-llama-cpp or --llama-cpp-dir PATH."
        )
    dest = Path("/kaggle/working/llama.cpp") if os.path.exists("/kaggle/working") else Path("llama.cpp")
    if not dest.exists():
        log.info("Cloning ggml-org/llama.cpp (shallow) into %s ...", dest)
        r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to clone llama.cpp: {r.stderr.strip()}")
    return dest


def convert_to_gguf(src_dir: Path, out_gguf: Path, outtype: str, is_lora: bool, llama_cpp_dir: Path) -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        log.info("Installing 'gguf' Python package for GGUF conversion ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to install gguf: {r.stderr.strip()}")

    script = llama_cpp_dir / ("convert_lora_to_gguf.py" if is_lora else "convert_hf_to_gguf.py")
    if not script.exists():
        raise FileNotFoundError(f"Converter script not found: {script}")

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(src_dir), "--outfile", str(out_gguf), "--outtype", outtype]
    if is_lora:
        cmd.extend(["--base-model-id", TrainConfig().model_id])
    log.info("Running GGUF conversion: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + "\n" + r.stderr).strip()[-1500:]
        raise RuntimeError(f"GGUF conversion failed (exit {r.returncode}):\n{tail}")
    log.info("GGUF model saved to %s", out_gguf)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--pt", action="store_true", help="also save PyTorch .pt state_dict")
    ap.add_argument("--merge", metavar="OUT_DIR")
    ap.add_argument("--gguf", metavar="OUT_FILE", help="convert merged model (or adapter if no --merge) to .gguf")
    ap.add_argument("--gguf-type", default="f16", help="GGUF outtype (e.g. f16, bf16, q8_0)")
    ap.add_argument("--llama-cpp-dir", help="path to llama.cpp repo")
    ap.add_argument("--fetch-llama-cpp", action="store_true", help="shallow-clone llama.cpp if not found")
    args = ap.parse_args(argv)
    setup_logging()

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        log.error("No adapter_config.json in %s", adapter)
        return 2
    if not any([args.zip, args.pt, args.merge, args.gguf]):
        ap.error("choose at least one of --zip, --pt, --merge OUT_DIR, --gguf OUT_FILE")

    if args.pt:
        save_adapter_pt(adapter)

    if args.zip:
        target = shutil.make_archive(str(adapter.parent / "gorphinai_adapter"), "zip", root_dir=adapter)
        log.info("Adapter archive: %s", target)

    merged_dir: Path | None = None
    if args.merge:
        import torch
        from peft import PeftModel

        from model_loader import load_base_model, load_tokenizer

        cfg = TrainConfig()
        cfg.use_qlora = False
        cfg.precision = "fp16" if torch.cuda.is_available() else "fp32"
        base = load_base_model(cfg, for_training=False, device_map="cpu" if not torch.cuda.is_available() else None)
        merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
        merged_dir = Path(args.merge)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        load_tokenizer(cfg.model_id).save_pretrained(str(merged_dir))
        if args.pt:
            pt_path = merged_dir / "pytorch_model.pt"
            torch.save(merged.state_dict(), str(pt_path))
            log.info("Merged PyTorch weights saved to %s", pt_path)
        log.info("Merged model saved to %s", merged_dir)

    if args.gguf:
        try:
            llama_dir = resolve_llama_cpp(args.llama_cpp_dir, args.fetch_llama_cpp)
            src = merged_dir if merged_dir else adapter
            convert_to_gguf(src, Path(args.gguf), args.gguf_type, is_lora=(merged_dir is None), llama_cpp_dir=llama_dir)
        except Exception as e:  # noqa: BLE001
            log.error("GGUF export failed: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
