"""Quality evaluation: BEFORE (base weights, adapter disabled) vs AFTER (with LoRA adapter).

  python evaluate.py --adapter outputs/adapter
  python evaluate.py --human-summary outputs/eval/human_scores.json     # after you fill in manual scores

Automatic metrics are simple, deterministic checks (language, expected answer, format) - they are a rough
signal, NOT a full measure of quality. Subjective quality is judged by you: the script writes review.md and
human_scores.json (score each answer 1-5 for base and tuned).  Greedy decoding is used for reproducibility.
Training loss alone is not used as a quality indicator.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from chat_utils import generate_reply
from config import TrainConfig, default_output_dir, setup_logging
from dataset_validator import detect_language

log = logging.getLogger("gorphinai")


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


# check kinds: lang, contains_any, regex, bullets_n, one_sentence, code
TESTS = [
    # --- languages
    {"id": "uk1", "group": "ukrainian", "lang": "uk", "messages": [U("Розкажи коротко, навіщо потрібен сон людині.")], "check": {"lang": "uk"}},
    {"id": "uk2", "group": "ukrainian", "lang": "uk", "messages": [U("Поясни, що таке гравітація, двома реченнями.")], "check": {"lang": "uk"}},
    {"id": "uk3", "group": "ukrainian", "lang": "uk", "messages": [U("Яка столиця України?")], "check": {"lang": "uk", "contains_any": ["Київ"]}},
    {"id": "ru1", "group": "russian", "lang": "ru", "messages": [U("Расскажи коротко, зачем человеку нужен сон.")], "check": {"lang": "ru"}},
    {"id": "ru2", "group": "russian", "lang": "ru", "messages": [U("Объясни, что такое гравитация, двумя предложениями.")], "check": {"lang": "ru"}},
    {"id": "ru3", "group": "russian", "lang": "ru", "messages": [U("Какая планета ближе всего к Солнцу?")], "check": {"lang": "ru", "contains_any": ["Меркурий"]}},
    {"id": "en1", "group": "english", "lang": "en", "messages": [U("Briefly, why do humans need sleep?")], "check": {"lang": "en"}},
    {"id": "en2", "group": "english", "lang": "en", "messages": [U("Explain gravity in two sentences.")], "check": {"lang": "en"}},
    {"id": "en3", "group": "english", "lang": "en", "messages": [U("What is the chemical symbol for gold?")], "check": {"lang": "en", "contains_any": ["Au"]}},
    # --- programming
    {"id": "code1", "group": "programming", "lang": "en", "messages": [U("Write a Python function that returns the largest number in a list.")], "check": {"code": True, "contains_any": ["def ", "max("]}},
    {"id": "code2", "group": "programming", "lang": "en", "messages": [U("Write a JavaScript arrow function that doubles every number in an array.")], "check": {"code": True, "contains_any": ["=>", "function", "map"]}},
    {"id": "code3", "group": "programming", "lang": "uk", "messages": [U("Напиши на Python цикл, що виводить числа від 1 до 5.")], "check": {"code": True, "contains_any": ["range", "for", "while"]}},
    # --- math
    {"id": "math1", "group": "math", "lang": "en", "messages": [U("What is 17 * 23? Give the final number.")], "check": {"regex": r"\b391\b"}},
    {"id": "math2", "group": "math", "lang": "ru", "messages": [U("Реши уравнение 4x + 6 = 30. В ответе укажи x.")], "check": {"regex": r"\b6\b"}},
    {"id": "math3", "group": "math", "lang": "uk", "messages": [U("Скільки буде 15% від 240?")], "check": {"regex": r"\b36\b"}},
    # --- logic
    {"id": "logic1", "group": "logic", "lang": "en", "messages": [U("Anna is taller than Ben, and Ben is taller than Carl. Who is the shortest?")], "check": {"contains_any": ["Carl"]}},
    {"id": "logic2", "group": "logic", "lang": "en", "messages": [U("What is the next number: 2, 4, 8, 16, ...?")], "check": {"regex": r"\b32\b"}},
    {"id": "logic3", "group": "logic", "lang": "ru", "messages": [U("Все птицы имеют перья. Пингвин — птица. Есть ли у пингвина перья?")], "check": {"contains_any": ["Да", "да"]}},
    # --- instruction following
    {"id": "if1", "group": "instructions", "lang": "en", "messages": [U("Give me exactly 3 bullet points about the benefits of exercise. Use '-' for bullets and write nothing else.")], "check": {"bullets_n": 3}},
    {"id": "if2", "group": "instructions", "lang": "en", "messages": [U("Answer in one sentence only: what is the Sun?")], "check": {"one_sentence": True}},
    {"id": "if3", "group": "instructions", "lang": "uk", "messages": [U("Відповідай лише одним словом: яка столиця Франції?")], "check": {"max_words": 3}},
    # --- multi-turn
    {"id": "mt1", "group": "multi_turn", "lang": "en", "messages": [U("Hi, my name is Dana and I love chess."), A("Nice to meet you, Dana! Chess is a great game."), U("What is my name and what do I love?")], "check": {"contains_all": ["Dana", "chess"], "case_insensitive": True}},
    {"id": "mt2", "group": "multi_turn", "lang": "uk", "messages": [U("Мене звати Оксана, я вивчаю біологію."), A("Приємно познайомитися, Оксано! Біологія — цікава наука."), U("Що я вивчаю?")], "check": {"contains_any": ["біолог", "Біолог"]}},
    {"id": "mt3", "group": "multi_turn", "lang": "ru", "messages": [U("Загадай число от 1 до 3 и запомни его: пусть это будет 2."), A("Хорошо, число 2 загадано."), U("Какое число ты загадал?")], "check": {"regex": r"\b2\b|двойк|два"}},
]


def check_answer(t: dict, text: str) -> dict:
    c, res = t["check"], {}
    low = text.lower()
    if "lang" in c:
        det = detect_language(text)
        want = c["lang"]
        res["language"] = det == want or (det == "cyrillic" and want in {"uk", "ru"})
    if "contains_any" in c:
        res["contains_any"] = any(k.lower() in low for k in c["contains_any"])
    if "contains_all" in c:
        res["contains_all"] = all(k.lower() in low for k in c["contains_all"])
    if "regex" in c:
        res["regex"] = re.search(c["regex"], text, re.I) is not None
    if c.get("code"):
        res["code_block_or_syntax"] = "```" in text or "    " in text or "=>" in text
    if "bullets_n" in c:
        n = len([l for l in text.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+\S", l)])
        res["bullets"] = n == c["bullets_n"]
    if c.get("one_sentence"):
        res["one_sentence"] = len(re.findall(r"[.!?](\s|$)", text.strip())) <= 1
    if "max_words" in c:
        res["max_words"] = len(text.split()) <= c["max_words"]
    words = text.split()
    res["no_degenerate_repetition"] = (len(set(words)) / max(1, len(words)) > 0.3) if len(words) > 15 else True
    res["non_empty"] = bool(text.strip())
    return res


def run_set(model, tok, tag: str, max_new_tokens: int) -> list[dict]:
    out = []
    for t in TESTS:
        t0 = time.time()
        try:
            text = generate_reply(model, tok, t["messages"], t["lang"], max_new_tokens=max_new_tokens, temperature=0.0,
                                  repetition_penalty=1.0)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] %s failed: %s", tag, t["id"], e)
            text = ""
        checks = check_answer(t, text)
        out.append({"id": t["id"], "group": t["group"], "response": text, "checks": checks,
                    "passed": all(checks.values()), "seconds": round(time.time() - t0, 2)})
        log.info("[%s] %-6s %s", tag, t["id"], "PASS" if out[-1]["passed"] else "FAIL")
    return out


def summarize(results: list[dict]) -> dict:
    g = defaultdict(list)
    for r in results:
        g[r["group"]].append(r["passed"])
    s = {k: round(sum(v) / len(v), 3) for k, v in g.items()}
    s["OVERALL"] = round(sum(r["passed"] for r in results) / max(1, len(results)), 3)
    return s


def human_summary(path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for side in ("base", "tuned"):
        sc = [d[f"{side}_score"] for d in data if isinstance(d.get(f"{side}_score"), (int, float))]
        print(f"{side}: n={len(sc)} mean={sum(sc)/len(sc):.2f}" if sc else f"{side}: no scores filled in yet")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="path to trained adapter dir (contains adapter_config.json)")
    ap.add_argument("--out-dir", default=str(Path(default_output_dir()) / "eval"))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (for small GPUs)")
    ap.add_argument("--human-summary")
    args = ap.parse_args(argv)
    setup_logging()
    if args.human_summary:
        return human_summary(args.human_summary)
    if not args.adapter:
        ap.error("--adapter is required")
    if args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be >= 1")

    from model_loader import load_with_adapter

    cfg = TrainConfig()
    cfg.use_qlora = args.qlora
    model, tok = load_with_adapter(args.adapter, cfg)
    with model.disable_adapter():
        before = run_set(model, tok, "BASE", args.max_new_tokens)
    after = run_set(model, tok, "TUNED", args.max_new_tokens)

    sb, sa = summarize(before), summarize(after)
    print("\n{:<14}{:>8}{:>8}{:>8}".format("group", "base", "tuned", "delta"))
    for k in sa:
        print("{:<14}{:>8.2f}{:>8.2f}{:>+8.2f}".format(k, sb[k], sa[k], sa[k] - sb[k]))
    print(f"\nNote: only {len(TESTS)} tests - differences of one test are noise. Read review.md as well.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"base": before, "tuned": after, "summary_base": sb, "summary_tuned": sa},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    prompts = {t["id"]: t["messages"][-1]["content"] for t in TESTS}
    with open(out / "review.md", "w", encoding="utf-8") as f:
        for b, a in zip(before, after):
            f.write(f"## {b['id']} ({b['group']})\n**Prompt:** {prompts[b['id']]}\n\n**Base:** {b['response']}\n\n**Tuned:** {a['response']}\n\n---\n")
    (out / "human_scores.json").write_text(json.dumps(
        [{"id": b["id"], "base_score": None, "tuned_score": None, "comment": ""} for b in before], indent=2), encoding="utf-8")
    log.info("Saved results to %s (fill human_scores.json with 1-5 scores, then use --human-summary)", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
