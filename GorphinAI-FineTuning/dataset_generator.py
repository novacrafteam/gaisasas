"""GorphinAI dataset generator (uk / ru / en).

Modes
  template : no external API. Hand-written banks + parametric generators (math/logic answers are COMPUTED).
  ai       : optional. Uses an OpenAI-compatible chat endpoint YOU provide (--api-base, --model, key in the
             GORPHIN_API_KEY environment variable). No key or paid API is bundled.

All output is synthetic and is labelled in the `source` field. Run:
  python dataset_generator.py --count 600 --out data/all.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path

import data_banks as B
from config import SYSTEM_PROMPTS, setup_logging

log = logging.getLogger("gorphinai")
LANGS = ["uk", "ru", "en"]
CATEGORIES = ["chat", "programming", "python", "javascript", "math", "logic", "science", "technology",
              "education", "text_work", "explain", "analysis", "problem_solving", "multi_turn", "honesty"]


# ----------------------------------------------------------------------------- helpers
def rec(lang: str, cat: str, msgs: list[tuple[str, str]], source: str = "template") -> dict:
    messages = []
    for u, a in msgs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return {"language": lang, "category": cat, "source": source, "messages": messages}


def pick_static(rng, bank, lang):
    return rng.choice(bank)[lang]


# ----------------------------------------------------------------------------- code
def gen_code(rng, lang, tasks, req):
    task = rng.choice(tasks)
    q = rng.choice(req[lang]).format(t=task["t"][lang])
    fence = "python" if tasks is B.PY_TASKS else "javascript"
    intro = {"uk": "Ось рішення:", "ru": "Вот решение:", "en": "Here is a solution:"}[lang]
    a = f"{intro}\n\n```{fence}\n{task['code']}\n```\n\n{task['x'][lang]}"
    return q, a


# ----------------------------------------------------------------------------- math
def m_linear(rng, lang):
    x = rng.randint(-12, 15)
    a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
    b = rng.randint(-20, 20)
    c = a * x + b
    sgn = "+" if b >= 0 else "-"
    eq = f"{a}x {sgn} {abs(b)} = {c}"
    q = {"uk": f"Розв'яжи рівняння: {eq}", "ru": f"Реши уравнение: {eq}", "en": f"Solve the equation: {eq}"}[lang]
    rhs = c - b
    steps = {
        "uk": f"1) Переносимо вільний член: {a}x = {c} - ({b}) = {rhs}.\n2) Ділимо на {a}: x = {rhs} / {a} = {x}.\nПеревірка: {a}·({x}) {sgn} {abs(b)} = {c} ✓",
        "ru": f"1) Переносим свободный член: {a}x = {c} - ({b}) = {rhs}.\n2) Делим на {a}: x = {rhs} / {a} = {x}.\nПроверка: {a}·({x}) {sgn} {abs(b)} = {c} ✓",
        "en": f"1) Move the constant: {a}x = {c} - ({b}) = {rhs}.\n2) Divide by {a}: x = {rhs} / {a} = {x}.\nCheck: {a}·({x}) {sgn} {abs(b)} = {c} ✓",
    }[lang]
    return q, steps + {"uk": f"\n\nВідповідь: x = {x}.", "ru": f"\n\nОтвет: x = {x}.", "en": f"\n\nAnswer: x = {x}."}[lang]


def m_percent(rng, lang):
    n = rng.choice([40, 60, 80, 120, 150, 200, 240, 300, 450, 500, 800, 1000])
    p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
    res = n * p / 100
    r = f"{res:g}"
    q = {"uk": f"Скільки становить {p}% від {n}?", "ru": f"Сколько составляет {p}% от {n}?", "en": f"What is {p}% of {n}?"}[lang]
    a = {"uk": f"{p}% від {n} = {n} · {p} / 100 = {r}.", "ru": f"{p}% от {n} = {n} · {p} / 100 = {r}.", "en": f"{p}% of {n} = {n} · {p} / 100 = {r}."}[lang]
    return q, a


def m_rect(rng, lang):
    w, h = rng.randint(2, 30), rng.randint(2, 30)
    q = {"uk": f"Прямокутник має сторони {w} см і {h} см. Знайди периметр і площу.",
         "ru": f"Прямоугольник имеет стороны {w} см и {h} см. Найди периметр и площадь.",
         "en": f"A rectangle has sides of {w} cm and {h} cm. Find its perimeter and area."}[lang]
    a = {"uk": f"Периметр P = 2·({w} + {h}) = {2*(w+h)} см.\nПлоща S = {w}·{h} = {w*h} см².",
         "ru": f"Периметр P = 2·({w} + {h}) = {2*(w+h)} см.\nПлощадь S = {w}·{h} = {w*h} см².",
         "en": f"Perimeter P = 2·({w} + {h}) = {2*(w+h)} cm.\nArea S = {w}·{h} = {w*h} cm²."}[lang]
    return q, a


def m_speed(rng, lang):
    v, t = rng.choice([40, 50, 60, 72, 80, 90, 100, 110]), rng.choice([1.5, 2, 2.5, 3, 4, 5])
    d = v * t
    q = {"uk": f"Автомобіль їде зі швидкістю {v} км/год протягом {t:g} год. Яку відстань він проїде?",
         "ru": f"Автомобиль едет со скоростью {v} км/ч в течение {t:g} ч. Какое расстояние он проедет?",
         "en": f"A car travels at {v} km/h for {t:g} hours. How far does it go?"}[lang]
    a = {"uk": f"Відстань = швидкість · час = {v} · {t:g} = {d:g} км.",
         "ru": f"Расстояние = скорость · время = {v} · {t:g} = {d:g} км.",
         "en": f"Distance = speed · time = {v} · {t:g} = {d:g} km."}[lang]
    return q, a


def m_avg(rng, lang):
    while True:
        nums = [rng.randint(1, 50) for _ in range(4)]
        if sum(nums) % 4 == 0:
            break
    s = ", ".join(map(str, nums))
    q = {"uk": f"Знайди середнє арифметичне чисел: {s}.", "ru": f"Найди среднее арифметическое чисел: {s}.",
         "en": f"Find the mean of these numbers: {s}."}[lang]
    a = {"uk": f"Сума = {sum(nums)}, кількість = 4, середнє = {sum(nums)} / 4 = {sum(nums)//4}.",
         "ru": f"Сумма = {sum(nums)}, количество = 4, среднее = {sum(nums)} / 4 = {sum(nums)//4}.",
         "en": f"Sum = {sum(nums)}, count = 4, mean = {sum(nums)} / 4 = {sum(nums)//4}."}[lang]
    return q, a


def m_gcd(rng, lang):
    g = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12])
    a, b = g * rng.randint(2, 15), g * rng.randint(2, 15)
    q = {"uk": f"Знайди НСД чисел {a} і {b}.", "ru": f"Найди НОД чисел {a} и {b}.", "en": f"Find the GCD of {a} and {b}."}[lang]
    res = math.gcd(a, b)
    ans = {"uk": f"За алгоритмом Евкліда ділимо більше число на менше й беремо остачі, поки не отримаємо 0. Останній ненульовий залишок — це НСД: НСД({a}, {b}) = {res}.",
           "ru": f"По алгоритму Евклида делим большее число на меньшее и берём остатки, пока не получим 0. Последний ненулевой остаток — это НОД: НОД({a}, {b}) = {res}.",
           "en": f"Using the Euclidean algorithm, divide the larger number by the smaller and keep the remainders until you reach 0. The last non-zero remainder is the GCD: GCD({a}, {b}) = {res}."}[lang]
    return q, ans


def m_frac(rng, lang):
    b, d = rng.choice([2, 3, 4, 5, 6, 8]), rng.choice([2, 3, 4, 5, 6, 8])
    a, c = rng.randint(1, b - 1) if b > 1 else 1, rng.randint(1, d - 1) if d > 1 else 1
    s = Fraction(a, b) + Fraction(c, d)
    lcm = b * d // math.gcd(b, d)
    num = a * (lcm // b) + c * (lcm // d)
    q = {"uk": f"Обчисли: {a}/{b} + {c}/{d}", "ru": f"Вычисли: {a}/{b} + {c}/{d}", "en": f"Compute: {a}/{b} + {c}/{d}"}[lang]
    simp = f"{s.numerator}/{s.denominator}" + (f" = {s.numerator//s.denominator} {s.numerator%s.denominator}/{s.denominator}" if s > 1 and s.denominator != 1 else "")
    if s.denominator == 1:
        simp = str(s.numerator)
    a_txt = {"uk": f"Спільний знаменник — {lcm}: {a*(lcm//b)}/{lcm} + {c*(lcm//d)}/{lcm} = {num}/{lcm}. Скорочуємо: {simp}.",
             "ru": f"Общий знаменатель — {lcm}: {a*(lcm//b)}/{lcm} + {c*(lcm//d)}/{lcm} = {num}/{lcm}. Сокращаем: {simp}.",
             "en": f"Common denominator is {lcm}: {a*(lcm//b)}/{lcm} + {c*(lcm//d)}/{lcm} = {num}/{lcm}. Simplified: {simp}."}[lang]
    return q, a_txt


def m_discount(rng, lang):
    price = rng.choice([200, 250, 400, 500, 800, 1200, 1500, 2000])
    d = rng.choice([10, 20, 25, 30, 40, 50])
    final = price * (100 - d) / 100
    q = {"uk": f"Товар коштував {price} грн, знижка {d}%. Яка нова ціна?", "ru": f"Товар стоил {price} руб., скидка {d}%. Какая новая цена?",
         "en": f"An item cost {price} and has a {d}% discount. What's the new price?"}[lang]
    a = {"uk": f"Знижка = {price}·{d}/100 = {price*d/100:g} грн. Нова ціна = {price} - {price*d/100:g} = {final:g} грн.",
         "ru": f"Скидка = {price}·{d}/100 = {price*d/100:g} руб. Новая цена = {price} - {price*d/100:g} = {final:g} руб.",
         "en": f"Discount = {price}·{d}/100 = {price*d/100:g}. New price = {price} - {price*d/100:g} = {final:g}."}[lang]
    return q, a


MATH_GENS = [m_linear, m_percent, m_rect, m_speed, m_avg, m_gcd, m_frac, m_discount]


# ----------------------------------------------------------------------------- logic
DAYS = {"uk": ["понеділок", "вівторок", "середа", "четвер", "пʼятниця", "субота", "неділя"],
        "ru": ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"],
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
SYLL = [  # (all X are Y) in three languages: X plural, Y plural, member noun phrase
    {"uk": ("кішки", "ссавці", "кішка"), "ru": ("кошки", "млекопитающие", "кошка"), "en": ("cats", "mammals", "a cat")},
    {"uk": ("троянди", "рослини", "троянда"), "ru": ("розы", "растения", "роза"), "en": ("roses", "plants", "a rose")},
    {"uk": ("квадрати", "прямокутники", "квадрат"), "ru": ("квадраты", "прямоугольники", "квадрат"), "en": ("squares", "rectangles", "a square")},
]


def l_order(rng, lang):
    names = rng.sample(B.NAMES[lang], 3)  # names[0] is the greatest ... names[2] the least
    kind = rng.choice(["height", "age"])
    A, Bn, C = names
    tpl = {
        ("height", "uk"): ("{x} має більший зріст, ніж {y}", "Хто з них найвищий за зростом?", "Найбільший зріст має {A}"),
        ("height", "ru"): ("{x} имеет больший рост, чем {y}", "Кто из них самый высокий?", "Самый высокий — {A}"),
        ("height", "en"): ("{x} is taller than {y}", "Who is the tallest?", "{A} is the tallest"),
        ("age", "uk"): ("{x} має більший вік, ніж {y}", "Хто з них наймолодший?", "Наймолодший — {C}"),
        ("age", "ru"): ("{x} имеет больший возраст, чем {y}", "Кто из них самый младший?", "Самый младший — {C}"),
        ("age", "en"): ("{x} is older than {y}", "Who is the youngest?", "{C} is the youngest"),
    }[(kind, lang)]
    f1, f2 = tpl[0].format(x=A, y=Bn), tpl[0].format(x=Bn, y=C)
    joiner = {"uk": "а", "ru": "а", "en": "and"}[lang]
    q = f"{f1}, {joiner} {f2}. {tpl[1]}"
    ans = tpl[2].format(A=A, C=C)
    why = {"uk": f"Оскільки {A} > {Bn} і {Bn} > {C}, за транзитивністю {A} > {C}. ",
           "ru": f"Так как {A} > {Bn} и {Bn} > {C}, по транзитивности {A} > {C}. ",
           "en": f"Since {A} > {Bn} and {Bn} > {C}, by transitivity {A} > {C}. "}[lang]
    return q, why + ans + "."

def l_seq(rng, lang):
    if rng.random() < 0.5:
        a0, d = rng.randint(1, 20), rng.randint(2, 9)
        seq = [a0 + i * d for i in range(5)]
        nxt = a0 + 5 * d
        rule = {"uk": f"кожне число більше за попереднє на {d}", "ru": f"каждое число больше предыдущего на {d}", "en": f"each number is {d} more than the previous one"}[lang]
    else:
        a0, r = rng.randint(1, 5), rng.choice([2, 3])
        seq = [a0 * r ** i for i in range(5)]
        nxt = a0 * r ** 5
        rule = {"uk": f"кожне число більше за попереднє в {r} рази", "ru": f"каждое число больше предыдущего в {r} раза", "en": f"each number is {r} times the previous one"}[lang]
    s = ", ".join(map(str, seq))
    q = {"uk": f"Яке наступне число в послідовності: {s}, ...?", "ru": f"Какое следующее число в последовательности: {s}, ...?", "en": f"What is the next number in the sequence: {s}, ...?"}[lang]
    a = {"uk": f"Закономірність: {rule}. Наступне число — {nxt}.", "ru": f"Закономерность: {rule}. Следующее число — {nxt}.", "en": f"Pattern: {rule}. The next number is {nxt}."}[lang]
    return q, a


def l_syll(rng, lang):
    X, Y, member = rng.choice(SYLL)[lang]
    name = {"uk": "Об'єкт", "ru": "Объект", "en": "Object"}[lang] + " " + rng.choice("ABCDEFGH")
    valid = rng.random() < 0.5
    if valid:
        q = {"uk": f"Усі {X} — це {Y}. {name} — це {member}. Чи є {name} одним із {Y}?",
             "ru": f"Все {X} — это {Y}. {name} — это {member}. Является ли {name} одним из {Y}?",
             "en": f"All {X} are {Y}. {name} is {member}. Is {name} one of the {Y}?"}[lang]
        a = {"uk": f"Так. Якщо всі {X} належать до {Y}, а {name} — {member}, то й {name} належить до {Y}.",
             "ru": f"Да. Если все {X} относятся к {Y}, а {name} — {member}, то и {name} относится к {Y}.",
             "en": f"Yes. If all {X} belong to {Y} and {name} is {member}, then {name} belongs to {Y} too."}[lang]
    else:
        q = {"uk": f"Усі {X} — це {Y}. {name} належить до {Y}. Чи випливає з цього, що {name} — це {member}?",
             "ru": f"Все {X} — это {Y}. {name} относится к {Y}. Следует ли из этого, что {name} — это {member}?",
             "en": f"All {X} are {Y}. {name} is one of the {Y}. Does it follow that {name} is {member}?"}[lang]
        a = {"uk": f"Ні, не обов'язково. З «усі {X} — {Y}» не випливає «усі {Y} — {X}». {name} може належати до {Y}, не будучи {member}.",
             "ru": f"Нет, не обязательно. Из «все {X} — {Y}» не следует «все {Y} — {X}». {name} может относиться к {Y}, не будучи {member}.",
             "en": f"No, not necessarily. \"All {X} are {Y}\" does not mean \"all {Y} are {X}\". {name} can be one of the {Y} without being {member}."}[lang]
    return q, a


def l_day(rng, lang):
    i, n = rng.randrange(7), rng.randint(10, 100)
    j = (i + n) % 7
    d = DAYS[lang]
    q = {"uk": f"Сьогодні {d[i]}. Який день тижня буде через {n} дн.?", "ru": f"Сегодня {d[i]}. Какой день недели будет через {n} дн.?",
         "en": f"Today is {d[i]}. What day of the week will it be in {n} days?"}[lang]
    a = {"uk": f"Тиждень має 7 днів, тож рахуємо остачу: {n} mod 7 = {n % 7}. Зсув уперед на {n % 7} дн.: {d[i]} → {d[j]}.",
         "ru": f"В неделе 7 дней, поэтому считаем остаток: {n} mod 7 = {n % 7}. Сдвиг вперёд на {n % 7} дн.: {d[i]} → {d[j]}.",
         "en": f"A week has 7 days, so take the remainder: {n} mod 7 = {n % 7}. Moving {n % 7} day(s) forward from {d[i]} gives {d[j]}."}[lang]
    return q, a


LOGIC_GENS = [l_order, l_seq, l_syll, l_day]

STATIC = {
    "chat": B.CHAT, "programming": B.PROGRAMMING, "science": B.SCIENCE, "technology": B.TECH, "education": B.EDUCATION,
    "text_work": B.TEXT_WORK, "explain": B.EXPLAIN, "analysis": B.ANALYSIS, "problem_solving": B.PROBLEM, "honesty": B.HONESTY,
}
BRIDGE = {"uk": ["А ще питання: ", "Дякую. Тепер інше: ", ""], "ru": ["А ещё вопрос: ", "Спасибо. Теперь другое: ", ""], "en": ["One more question: ", "Thanks. Now something else: ", ""]}


def one_turn(rng, lang, cat) -> tuple[str, str]:
    if cat == "python":
        return gen_code(rng, lang, B.PY_TASKS, B.PY_REQ)
    if cat == "javascript":
        return gen_code(rng, lang, B.JS_TASKS, B.JS_REQ)
    if cat == "math":
        return rng.choice(MATH_GENS)(rng, lang)
    if cat == "logic":
        return rng.choice(LOGIC_GENS)(rng, lang)
    return pick_static(rng, STATIC[cat], lang)


def make_example(rng, lang, cat):
    if cat == "multi_turn":
        pool = [c for c in CATEGORIES if c != "multi_turn"]
        n_turns = rng.choice([2, 2, 3])
        cats = rng.sample(pool, n_turns)
        turns = []
        for k, c in enumerate(cats):
            q, a = one_turn(rng, lang, c)
            if k > 0:
                q = rng.choice(BRIDGE[lang]) + q
            turns.append((q, a))
        return rec(lang, cat, turns)
    q, a = one_turn(rng, lang, cat)
    return rec(lang, cat, [(q, a)])


# ----------------------------------------------------------------------------- AI-assisted
AI_PROMPT = (
    "Create one realistic training dialogue for a friendly, calm, honest AI assistant named GorphinAI. "
    "Language: {lang_name}. Category: {cat}. Topic seed: {seed}. "
    "Rules: be factually correct; if unsure the assistant must say so; no placeholders. "
    'Return ONLY JSON: {{"messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}} '
    "with 1-3 user/assistant exchanges, ending with an assistant message."
)
LANG_NAME = {"uk": "Ukrainian", "ru": "Russian", "en": "English"}


def ai_example(rng, lang, cat, api_base, model, key):
    seed = rng.randint(1, 10**6)
    body = json.dumps({"model": model, "temperature": 0.9, "messages": [
        {"role": "user", "content": AI_PROMPT.format(lang_name=LANG_NAME[lang], cat=cat, seed=seed)}]}).encode()
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        text = text[text.index("{"): text.rindex("}") + 1]
        msgs = json.loads(text)["messages"]
        return {"language": lang, "category": cat, "source": f"ai-assisted:{model}", "messages": msgs}
    except (urllib.error.URLError, KeyError, ValueError, IndexError, TimeoutError) as e:
        log.warning("AI generation failed (%s); skipping one sample", e)
        return None


# ----------------------------------------------------------------------------- main
def key_of(r: dict) -> str:
    return json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)


def load_existing(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                log.warning("Skipping corrupted line in existing file")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=600, help="target total number of examples")
    ap.add_argument("--out", default="data/all.jsonl")
    ap.add_argument("--mode", choices=["template", "ai"], default="template")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue: keep existing records, generate only the missing ones")
    ap.add_argument("--api-base", default=os.environ.get("GORPHIN_API_BASE"))
    ap.add_argument("--model", default=os.environ.get("GORPHIN_API_MODEL"))
    ap.add_argument("--max-share", type=float, default=0.15, help="max fraction of one language taken by a single category")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args(argv)
    setup_logging()

    if not 0.05 <= args.max_share <= 1:
        ap.error("--max-share must be in [0.05, 1]")
    if args.count < 3:
        ap.error("--count must be >= 3")
    key = os.environ.get("GORPHIN_API_KEY")
    if args.mode == "ai" and not (args.api_base and args.model and key):
        ap.error("AI mode needs --api-base, --model and the GORPHIN_API_KEY environment variable")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + (len(load_existing(out)) if args.resume else 0))
    records = load_existing(out) if args.resume else []
    seen = {key_of(r) for r in records}
    per_lang_target = {l: args.count // 3 + (1 if i < args.count % 3 else 0) for i, l in enumerate(LANGS)}
    per_lang = Counter(r["language"] for r in records)

    fails = Counter()
    attempts, max_attempts = 0, args.count * 60
    while any(per_lang[l] < per_lang_target[l] for l in LANGS) and attempts < max_attempts:
        attempts += 1
        lang = min(LANGS, key=lambda l: per_lang[l] / per_lang_target[l])
        if per_lang[lang] >= per_lang_target[lang]:
            continue
        counts = Counter(r["category"] for r in records if r["language"] == lang)
        open_cats = [c for c in CATEGORIES if fails[(lang, c)] < 40 and counts[c] < max(1, args.max_share * per_lang_target[lang])]  # skip categories out of unique samples
        if not open_cats:
            per_lang_target[lang] = per_lang[lang]
            continue
        cat = min(open_cats, key=lambda c: (counts[c], rng.random()))  # keep categories balanced
        ex = make_example(rng, lang, cat) if args.mode == "template" else ai_example(rng, lang, cat, args.api_base, args.model, key)
        if ex is None:
            continue
        total = sum(len(m["content"]) for m in ex["messages"])
        if not (args.min_chars <= total <= args.max_chars) or any(not m["content"].strip() for m in ex["messages"]):
            continue
        k = key_of(ex)
        if k in seen:
            fails[(lang, cat)] += 1
            continue
        seen.add(k)
        ex["id"] = f"{lang}-{cat}-{len(records):06d}"
        records.append(ex)
        per_lang[lang] += 1

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Wrote %d records to %s (requested %d)", len(records), out, args.count)
    log.info("By language: %s", dict(Counter(r["language"] for r in records)))
    log.info("By category: %s", dict(Counter(r["category"] for r in records)))
    if len(records) < args.count:
        log.warning("Template mode ran out of unique examples (%d of %d). Use --mode ai or extend data_banks.py.",
                    len(records), args.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
