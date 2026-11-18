"""Dataset validator: structure, emptiness, length, duplicates, language sanity, secrets/placeholders.

  python dataset_validator.py data/all.jsonl                 # report only
  python dataset_validator.py data/all.jsonl --fix --out data/clean.jsonl
Exit code 1 if invalid records were found (and --fix was not used).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from config import setup_logging

log = logging.getLogger("gorphinai")
ROLES = {"user", "assistant", "system"}
UK_ONLY = set("іїєґІЇЄҐ")
RU_ONLY = set("ыэъёЫЭЪЁ")
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})")
PLACEHOLDER_RE = re.compile(r"(lorem ipsum|\bTODO\b|\{\{.*?\}\}|<placeholder>|\[INSERT)", re.I)


def detect_language(text: str) -> str:
    cyr = sum("а" <= c.lower() <= "я" or c in "іїєґёъыэ" for c in text)
    lat = sum("a" <= c.lower() <= "z" for c in text)
    if cyr == 0 and lat == 0:
        return "unknown"
    if lat > cyr:
        return "en"
    if any(c in UK_ONLY for c in text):
        return "uk"
    if any(c in RU_ONLY for c in text):
        return "ru"
    return "cyrillic"  # ambiguous (short text with neither marker)


def check_record(r: dict, min_chars: int, max_chars: int) -> list[str]:
    errs: list[str] = []
    msgs = r.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["no messages list"]
    if r.get("language") not in {"uk", "ru", "en"}:
        errs.append(f"bad language field: {r.get('language')!r}")
    expected = "user"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            errs.append(f"message {i}: malformed")
            continue
        if m["role"] == "system":
            if i != 0:
                errs.append("system message must be first")
            continue
        if m["role"] != expected:
            errs.append(f"message {i}: expected role {expected}")
        expected = "assistant" if expected == "user" else "user"
        if not m["content"].strip():
            errs.append(f"message {i}: empty content")
        if SECRET_RE.search(m["content"]):
            errs.append(f"message {i}: looks like a secret/API key")
        if PLACEHOLDER_RE.search(m["content"]):
            errs.append(f"message {i}: placeholder text")
    if msgs[-1].get("role") != "assistant":
        errs.append("last message must be from assistant")
    total = sum(len(m.get("content", "")) for m in msgs if isinstance(m, dict))
    if total < min_chars:
        errs.append(f"too short ({total} chars)")
    if total > max_chars:
        errs.append(f"too long ({total} chars)")
    if not errs and r.get("language") in {"uk", "ru", "en"}:
        a = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        det = detect_language(a)
        # code-only answers may be mostly latin, so only flag clear contradictions on prose answers
        if r["language"] == "en" and det in {"uk", "ru"}:
            errs.append(f"language mismatch: labelled en, text looks {det}")
        if r["language"] == "uk" and det == "ru":
            errs.append("language mismatch: labelled uk, text looks ru")
        if r["language"] == "ru" and det == "uk":
            errs.append("language mismatch: labelled ru, text looks uk")
    return errs


def validate_file(path: Path, min_chars=2, max_chars=8000):
    good, bad, dup = [], [], 0
    seen = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            bad.append((n, [f"invalid JSON: {e}"]))
            continue
        errs = check_record(r, min_chars, max_chars)
        if errs:
            bad.append((n, errs))
            continue
        k = json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        good.append(r)
    return good, bad, dup


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--fix", action="store_true", help="write cleaned records (needs --out)")
    ap.add_argument("--out")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args(argv)
    setup_logging()
    p = Path(args.file)
    if not p.exists():
        log.error("File not found: %s", p)
        return 2
    if args.fix and not args.out:
        ap.error("--fix requires --out")
    good, bad, dup = validate_file(p, args.min_chars, args.max_chars)
    log.info("valid=%d invalid=%d exact_duplicates_removed=%d", len(good), len(bad), dup)
    for n, errs in bad[:20]:
        log.warning("line %d: %s", n, "; ".join(errs))
    if len(bad) > 20:
        log.warning("... and %d more invalid records", len(bad) - 20)
    log.info("languages: %s", dict(Counter(r["language"] for r in good)))
    log.info("categories: %s", dict(Counter(r.get("category", "?") for r in good)))
    if args.fix:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Cleaned dataset written to %s", args.out)
        return 0
    return 1 if (bad or dup) else 0


if __name__ == "__main__":
    sys.exit(main())
