"""Speed and memory benchmark (real measurements on your hardware, nothing simulated).

  python benchmark.py --adapter outputs/adapter                # inference tokens/s + peak VRAM
  python benchmark.py --train-step --preset local              # one real forward/backward step: time + peak VRAM
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from config import TrainConfig, preset, setup_logging

log = logging.getLogger("gorphinai")


def bench_inference(adapter, qlora, new_tokens, runs):
    import torch

    from chat_utils import generate_reply
    from model_loader import load_base_model, load_tokenizer, load_with_adapter

    cfg = TrainConfig()
    cfg.use_qlora = qlora
    if adapter:
        model, tok = load_with_adapter(adapter, cfg)
    else:
        tok = load_tokenizer(cfg.model_id)
        model = load_base_model(cfg, for_training=False)
        model.eval()
    msgs = [{"role": "user", "content": "Explain how a binary search works."}]
    generate_reply(model, tok, msgs, "en", max_new_tokens=8)  # warm-up
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rates = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        text = generate_reply(model, tok, msgs, "en", max_new_tokens=new_tokens, temperature=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        rates.append(n / (time.time() - t0))
    log.info("Inference: %.1f tokens/s (mean of %d runs: %s)", sum(rates) / len(rates), runs, [round(r, 1) for r in rates])
    if torch.cuda.is_available():
        log.info("Peak VRAM allocated: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)


def bench_train_step(cfg: TrainConfig):
    import torch

    from model_loader import attach_lora, load_base_model, resolve_precision
    from train import make_collator

    if not torch.cuda.is_available():
        log.error("No CUDA GPU: training-step benchmark needs one")
        return
    model = load_base_model(cfg, for_training=True)
    if cfg.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = attach_lora(model, cfg)
    _, fp16, bf16 = resolve_precision(cfg.precision)
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    L, B = cfg.max_seq_length, cfg.per_device_train_batch_size
    ids = torch.randint(1000, 5000, (B, L), device="cuda")
    batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16 if fp16 else torch.bfloat16, enabled=fp16 or bf16):
                loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    except torch.OutOfMemoryError:
        log.error("OOM at seq_len=%d batch=%d - this configuration does not fit on this GPU.", L, B)
        return
    log.info("Train step (batch=%d, seq=%d, full-length synthetic tokens): %.2fs (steps: %s)", B, L, min(times[1:]),
             [round(t, 2) for t in times])
    log.info("Peak VRAM allocated: %.2f GB of %.1f GB", torch.cuda.max_memory_allocated() / 1e9,
             torch.cuda.get_device_properties(0).total_memory / 1e9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--train-step", action="store_true")
    ap.add_argument("--preset", default="local", choices=["local", "kaggle"])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)
    setup_logging()
    if args.new_tokens < 1 or args.runs < 1:
        ap.error("--new-tokens and --runs must be >= 1")
    if args.train_step:
        bench_train_step(preset(args.preset))
    else:
        bench_inference(args.adapter, args.qlora, args.new_tokens, args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
