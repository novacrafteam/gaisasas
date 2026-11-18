"""Stratified (language x category) train/validation split.

  python dataset_splitter.py data/all.jsonl --val-ratio 0.1
Writes data/train.jsonl and data/validation.jsonl (next to the input unless --out-dir is given).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import setup_logging

log = logging.getLogger("gorphinai")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    setup_logging()
    if not 0 < args.val_ratio < 0.5:
        ap.error("--val-ratio must be in (0, 0.5)")
    src = Path(args.file)
    if not src.exists():
        log.error("File not found: %s", src)
        return 2
    records = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if len(records) < 10:
        log.error("Need at least 10 records to split, got %d", len(records))
        return 2

    groups = defaultdict(list)
    for r in records:
        groups[(r["language"], r.get("category", "?"))].append(r)
    rng = random.Random(args.seed)
    train, val = [], []
    for _, items in sorted(groups.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio)) if len(items) >= 3 else 0
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("validation.jsonl", val)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("train=%d validation=%d -> %s", len(train), len(val), out_dir)
    log.info("val languages: %s", dict(Counter(r["language"] for r in val)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
