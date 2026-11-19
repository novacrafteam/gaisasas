"""SFT with LoRA / QLoRA on Qwen/Qwen3.5-2B. The base weights are never modified.

Examples
  python train.py --preset local                 # GTX 1660 Ti: QLoRA, seq 512
  python train.py --preset kaggle                # Kaggle T4
  python train.py --preset local --resume        # continue from last checkpoint
  python train.py --preset cpu-test --dry-run    # no model download: checks data + config + memory estimate
Safe stop: create the file <output_dir>/STOP (the GUI does this) - a checkpoint is saved, then training ends.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from config import IS_KAGGLE, TrainConfig, preset, setup_logging

if IS_KAGGLE and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle has 2xT4; avoid accidental DataParallel

log = logging.getLogger("gorphinai")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="local", choices=["local", "kaggle", "cpu-test"])
    ap.add_argument("--config", help="JSON config saved by a previous run (overrides preset)")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir/checkpoints")
    ap.add_argument("--dry-run", action="store_true", help="validate everything except loading the model")
    ap.add_argument("--qlora", dest="qlora", action="store_true", default=None)
    ap.add_argument("--no-qlora", dest="qlora", action="store_false")
    for name, typ in [("train_file", str), ("validation_file", str), ("output_dir", str), ("max_seq_length", int),
                      ("lora_r", int), ("lora_alpha", int), ("per_device_train_batch_size", int),
                      ("gradient_accumulation_steps", int), ("learning_rate", float), ("num_train_epochs", float),
                      ("precision", str), ("dataloader_workers", int), ("eval_steps", int), ("save_steps", int),
                      ("model_id", str), ("revision", str)]:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=None)
    ap.add_argument("--export-merged", action="store_true", help="after training, merge adapter + save .safetensors & .pt")
    ap.add_argument("--export-gguf", action="store_true", help="after training, also export .gguf via llama.cpp")
    return ap.parse_args(argv)


def build_config(args) -> TrainConfig:
    cfg = TrainConfig.load(args.config) if args.config else preset(args.preset)
    for k, v in vars(args).items():
        if k in {"preset", "config", "resume", "dry_run", "export_merged", "export_gguf"} or v is None:
            continue
        if k == "qlora":
            cfg.use_qlora = v
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


# ----------------------------------------------------------------------------- data
def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}. Run dataset_generator.py + dataset_splitter.py first.")
    rows = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{n} is not valid JSON ({e}). Run dataset_validator.py") from e
    if not rows:
        raise ValueError(f"{p} is empty")
    return rows


class SFTDataset:
    """Tokenised examples. Loss only on the FINAL assistant message (earlier turns are context)."""

    def __init__(self, rows, tokenizer, max_len: int):
        from chat_utils import build_prompt

        self.items, skipped = [], 0
        for r in rows:
            msgs = r["messages"]
            prompt = build_prompt(tokenizer, msgs[:-1], r.get("language", "en"))
            answer = msgs[-1]["content"] + "<|im_end|>\n"
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(a_ids) > max_len:
                skipped += 1  # never silently truncate the answer; skip and report
                continue
            self.items.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
        if skipped:
            log.warning("Skipped %d examples longer than max_seq_length=%d", skipped, max_len)
        if not self.items:
            raise ValueError("No usable examples after tokenisation (increase --max-seq-length?)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), m), -100, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    return collate


# ----------------------------------------------------------------------------- callbacks
def make_callbacks(out_dir: Path, metrics_path: Path):
    from transformers import TrainerCallback

    from chat_utils import append_metric, gpu_stats, ram_stats

    stop_file = out_dir / "STOP"

    class Metrics(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.last_t, self.last_step = time.time(), 0

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            now = time.time()
            rec = {"event": "log", "step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch}
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if k in logs:
                    rec[k] = logs[k]
            dstep = state.global_step - self.last_step
            if dstep > 0:
                rec["sec_per_step"] = (now - self.last_t) / dstep
                self.last_t, self.last_step = now, state.global_step
            rec["elapsed_s"] = now - self.t0
            rec.update(gpu_stats())
            rec.update(ram_stats())
            append_metric(metrics_path, rec)

        def on_step_end(self, args, state, control, **kw):
            if stop_file.exists():
                log.warning("STOP file detected: saving checkpoint and stopping")
                control.should_save = True
                control.should_training_stop = True
            return control

    return [Metrics()]


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = build_config(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    from chat_utils import append_metric, latest_checkpoint

    metrics_path = out_dir / "metrics.jsonl"
    stop_file = out_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    cfg.save(out_dir / "train_config.json")
    log.info("Config: %s", json.dumps(cfg.__dict__, ensure_ascii=False))

    # data checks (no heavy imports needed)
    train_rows, val_rows = load_jsonl(cfg.train_file), load_jsonl(cfg.validation_file)
    log.info("Data: train=%d validation=%d", len(train_rows), len(val_rows))

    from model_loader import check_environment, estimate_memory_gb

    est = estimate_memory_gb(cfg)
    log.info("ROUGH VRAM estimate (GB): %s  (not a guarantee)", est)
    for w in check_environment():
        log.warning(w)
    if args.dry_run:
        log.info("Dry run finished: config, data and memory estimate OK. Model was not loaded.")
        return 0

    try:
        import torch
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as e:
        log.error("Missing dependency: %s. Run: pip install -r requirements.txt", e)
        return 2
    from model_loader import attach_lora, load_base_model, load_tokenizer, resolve_precision

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s, %.1f GB", torch.cuda.get_device_name(0), total)
        if est["total_rough"] > total * 0.95:
            log.warning("Estimated need (%.1f GB) is close to/above available VRAM (%.1f GB): expect OOM. "
                        "Lower --max-seq-length, enable --qlora, or lower LoRA rank.", est["total_rough"], total)
    else:
        log.warning("No CUDA GPU detected - training a 2B model on CPU is impractical (use --dry-run or a GPU).")

    set_seed(cfg.seed)
    append_metric(metrics_path, {"event": "stage", "stage": "loading_model"})
    tok = load_tokenizer(cfg.model_id, cfg.revision)
    append_metric(metrics_path, {"event": "stage", "stage": "tokenizing"})
    train_ds = SFTDataset(train_rows, tok, cfg.max_seq_length)
    val_ds = SFTDataset(val_rows, tok, cfg.max_seq_length)
    log.info("Tokenised: train=%d val=%d", len(train_ds), len(val_ds))

    model = load_base_model(cfg, for_training=True)
    _, fp16, bf16 = resolve_precision(cfg.precision)
    if cfg.use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = attach_lora(model, cfg)
    if fp16:  # GradScaler cannot unscale fp16 params -> keep trainable (LoRA) weights in fp32
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                p.data = p.data.float()

    ckpt_dir = out_dir / "checkpoints"
    import inspect
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup_kw = {"warmup_ratio": cfg.warmup_ratio}
    else:
        est_steps = max(1, len(train_ds) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps) * cfg.num_train_epochs)
        warmup_kw = {"warmup_steps": max(1, int(est_steps * cfg.warmup_ratio))} if cfg.warmup_ratio > 0 else {}
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        **warmup_kw,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        fp16=fp16, bf16=bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=cfg.dataloader_workers,
        optim="paged_adamw_8bit" if cfg.use_qlora else "adamw_torch",
        report_to="none", remove_unused_columns=False, seed=cfg.seed,
        gradient_checkpointing=False,  # already enabled on the model above
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id), callbacks=make_callbacks(out_dir, metrics_path))

    resume = None
    if args.resume:
        resume = latest_checkpoint(ckpt_dir)
        if resume is None:
            log.warning("--resume given but no checkpoint found in %s; starting fresh", ckpt_dir)
        else:
            log.info("Resuming from %s", resume)

    adapter_dir = out_dir / "adapter"
    append_metric(metrics_path, {"event": "stage", "stage": "training"})
    status = "done"
    try:
        trainer.train(resume_from_checkpoint=resume)
        if stop_file.exists():
            status = "stopped"
    except KeyboardInterrupt:
        status = "stopped"
        log.warning("Interrupted by user; saving adapter")
    except Exception as e:  # noqa: BLE001
        oom = isinstance(e, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()
        if oom:
            log.error("CUDA out of memory. Try: --max-seq-length 256, --lora-r 8, --qlora, batch size 1, "
                      "close other GPU programs. Last checkpoint (if any) is in %s", ckpt_dir)
            append_metric(metrics_path, {"event": "stage", "stage": "error", "message": "CUDA OOM"})
            return 3
        append_metric(metrics_path, {"event": "stage", "stage": "error", "message": str(e)})
        raise

    append_metric(metrics_path, {"event": "stage", "stage": "saving"})
    model.save_pretrained(str(adapter_dir))
    try:
        from peft import get_peft_model_state_dict

        torch.save(get_peft_model_state_dict(model), str(adapter_dir / "adapter_model.pt"))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not save adapter_model.pt: %s", e)
    tok.save_pretrained(str(adapter_dir))
    cfg.save(adapter_dir / "train_config.json")
    if status == "stopped":
        log.info("Stopped. Adapter + checkpoint saved. Continue later with --resume.")
    else:
        try:
            final = trainer.evaluate()
            append_metric(metrics_path, {"event": "log", "step": trainer.state.global_step,
                                         "max_steps": trainer.state.max_steps, "eval_loss": final.get("eval_loss")})
        except Exception as e:  # noqa: BLE001
            log.warning("Final evaluation failed: %s", e)
        if args.export_merged or args.export_gguf:
            try:
                import export_model

                ex_args = ["--adapter", str(adapter_dir), "--pt"]
                if args.export_merged or args.export_gguf:
                    ex_args += ["--merge", str(out_dir / "merged")]
                if args.export_gguf:
                    ex_args += ["--gguf", str(out_dir / "gorphinai.gguf"), "--fetch-llama-cpp"]
                export_model.main(ex_args)
            except Exception as e:  # noqa: BLE001
                log.warning("Extra export (.pt/.gguf) failed (adapter itself is safe in %s): %s", adapter_dir, e)
    append_metric(metrics_path, {"event": "stage", "stage": status})
    log.info("Adapter saved to %s", adapter_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
