"""Interactive tester: type ANY prompt, multi-turn chat, sampling controls, 👍/👎 feedback.

Local (Tkinter):   python chat_ui.py --adapter outputs/adapter
Kaggle/Jupyter:    from chat_ui import notebook_ui; notebook_ui("/kaggle/working/outputs/adapter")
Rated examples go to <output_dir>/feedback.jsonl. They are NEVER added to training automatically -
review them first, then copy the good ones into your dataset yourself.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

from chat_utils import generate_reply
from config import TrainConfig, default_output_dir

log = logging.getLogger("gorphinai")


class ChatSession:
    def __init__(self, adapter: str | None, use_qlora: bool = False, feedback_path: str | None = None):
        from model_loader import load_base_model, load_tokenizer, load_with_adapter

        cfg = TrainConfig()
        cfg.use_qlora = use_qlora
        if adapter:
            self.model, self.tok = load_with_adapter(adapter, cfg)
        else:
            self.tok = load_tokenizer(cfg.model_id)
            self.model = load_base_model(cfg, for_training=False)
            self.model.eval()
        self.messages: list[dict] = []
        self.feedback_path = Path(feedback_path or Path(default_output_dir()) / "feedback.jsonl")
        self.last_params: dict = {}

    def reset(self):
        self.messages = []

    def ask(self, text: str, lang: str = "en", max_new_tokens: int = 256, temperature: float = 0.7,
            top_p: float = 0.9, repetition_penalty: float = 1.05) -> str:
        if not text.strip():
            raise ValueError("Empty message")
        self.messages.append({"role": "user", "content": text})
        self.last_params = dict(lang=lang, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                repetition_penalty=repetition_penalty)
        try:
            reply = generate_reply(self.model, self.tok, self.messages, lang, max_new_tokens, temperature, top_p,
                                   repetition_penalty)
        except Exception:
            self.messages.pop()  # keep history consistent on failure
            raise
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def rate(self, good: bool) -> None:
        if len(self.messages) < 2 or self.messages[-1]["role"] != "assistant":
            raise ValueError("Nothing to rate yet")
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rating": "good" if good else "bad",
               "params": self.last_params, "messages": self.messages, "reviewed": False}
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- Tkinter
def run_tk(session_factory):
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("GorphinAI - test chat")
    root.geometry("900x700")
    state = {"session": None, "busy": False}

    log_box = tk.Text(root, wrap="word", state="disabled", height=24)
    log_box.pack(fill="both", expand=True, padx=8, pady=8)
    log_box.tag_config("user", foreground="#0b5cad")
    log_box.tag_config("bot", foreground="#1b7f3b")
    log_box.tag_config("sys", foreground="#888888")

    def write(tag, text):
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n\n", tag)
        log_box.see("end")
        log_box.configure(state="disabled")

    ctl = ttk.Frame(root)
    ctl.pack(fill="x", padx=8)
    lang = tk.StringVar(value="en")
    ttk.Label(ctl, text="Language").grid(row=0, column=0)
    ttk.Combobox(ctl, textvariable=lang, values=["uk", "ru", "en"], width=5, state="readonly").grid(row=0, column=1)
    vars_ = {}
    for i, (name, lo, hi, val, res) in enumerate([("Temperature", 0, 1.5, 0.7, 0.05), ("Top P", 0.1, 1.0, 0.9, 0.05),
                                                  ("Max New Tokens", 16, 1024, 256, 16), ("Repetition Penalty", 1.0, 1.5, 1.05, 0.01)]):
        v = tk.DoubleVar(value=val)
        vars_[name] = v
        ttk.Label(ctl, text=name).grid(row=0, column=2 + 2 * i, padx=(10, 0))
        tk.Scale(ctl, from_=lo, to=hi, resolution=res, orient="horizontal", variable=v, length=110).grid(row=0, column=3 + 2 * i)

    entry = tk.Text(root, height=3)
    entry.pack(fill="x", padx=8, pady=6)
    btns = ttk.Frame(root)
    btns.pack(fill="x", padx=8, pady=(0, 8))

    def send():
        s = state["session"]
        text = entry.get("1.0", "end").strip()
        if s is None or state["busy"] or not text:
            return
        state["busy"] = True
        entry.delete("1.0", "end")
        write("user", "You: " + text)
        params = dict(lang=lang.get(), max_new_tokens=int(vars_["Max New Tokens"].get()), temperature=vars_["Temperature"].get(),
                      top_p=vars_["Top P"].get(), repetition_penalty=vars_["Repetition Penalty"].get())

        def work():
            try:
                reply = s.ask(text, **params)
                root.after(0, lambda: write("bot", "GorphinAI: " + reply))
            except Exception as e:  # noqa: BLE001
                root.after(0, lambda: write("sys", f"[error] {e}"))
            finally:
                state["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    def rate(good):
        try:
            state["session"].rate(good)
            write("sys", "[saved to feedback.jsonl: " + ("good" if good else "bad") + "]")
        except Exception as e:  # noqa: BLE001
            messagebox.showinfo("Rate", str(e))

    def new_chat():
        if state["session"]:
            state["session"].reset()
        write("sys", "[new conversation]")

    ttk.Button(btns, text="Send", command=send).pack(side="left")
    ttk.Button(btns, text="👍 Good", command=lambda: rate(True)).pack(side="left", padx=6)
    ttk.Button(btns, text="👎 Bad", command=lambda: rate(False)).pack(side="left")
    ttk.Button(btns, text="New chat", command=new_chat).pack(side="left", padx=6)
    entry.bind("<Control-Return>", lambda e: (send(), "break")[1])

    write("sys", "Loading model... (the first time this downloads weights; please wait)")

    def load():
        try:
            state["session"] = session_factory()
            root.after(0, lambda: write("sys", "Model ready. Ctrl+Enter sends."))
        except Exception as e:  # noqa: BLE001
            root.after(0, lambda: write("sys", f"[load error] {e}"))

    threading.Thread(target=load, daemon=True).start()
    root.mainloop()


# ----------------------------------------------------------------------------- Jupyter / Kaggle
def notebook_ui(adapter: str | None, use_qlora: bool = False, feedback_path: str | None = None):
    """ipywidgets interface (preinstalled on Kaggle). Returns the ChatSession."""
    import ipywidgets as w
    from IPython.display import display

    s = ChatSession(adapter, use_qlora, feedback_path)
    out = w.Output(layout={"width": "100%"})
    out_box = w.Box([out], layout={"border": "1px solid #ccc", "height": "320px", "overflow_y": "auto", "width": "98%", "padding": "6px"})
    box = w.Textarea(placeholder="Type any message...", layout={"width": "98%", "height": "70px"})
    lang = w.Dropdown(options=["uk", "ru", "en"], value="en", description="Language")
    temp = w.FloatSlider(value=0.7, min=0, max=1.5, step=0.05, description="Temperature")
    topp = w.FloatSlider(value=0.9, min=0.1, max=1.0, step=0.05, description="Top P")
    mnt = w.IntSlider(value=256, min=16, max=1024, step=16, description="Max tokens")
    rep = w.FloatSlider(value=1.05, min=1.0, max=1.5, step=0.01, description="Rep. penalty")
    send, good, bad, new = (w.Button(description=d) for d in ["Send", "👍 Good", "👎 Bad", "New chat"])

    def on_send(_):
        text = box.value
        if not text.strip():
            return
        box.value = ""
        with out:
            print("You:", text)
            try:
                print("GorphinAI:", s.ask(text, lang.value, mnt.value, temp.value, topp.value, rep.value), "\n")
            except Exception as e:  # noqa: BLE001
                print("[error]", e)

    def on_rate(g):
        def f(_):
            with out:
                try:
                    s.rate(g)
                    print("[saved:", "good" if g else "bad", "]")
                except Exception as e:  # noqa: BLE001
                    print("[error]", e)
        return f

    def on_new(_):
        s.reset()
        with out:
            print("[new conversation]\n")

    send.on_click(on_send)
    good.on_click(on_rate(True))
    bad.on_click(on_rate(False))
    new.on_click(on_new)
    display(w.VBox([out_box, box, w.HBox([send, good, bad, new]), w.HBox([lang, temp]), w.HBox([topp, mnt, rep])]))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="adapter dir; omit to chat with the untouched base model")
    ap.add_argument("--qlora", action="store_true", help="load base in 4-bit (small GPUs)")
    ap.add_argument("--feedback", help="feedback jsonl path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_tk(lambda: ChatSession(args.adapter, args.qlora, args.feedback))
    return 0


if __name__ == "__main__":
    sys.exit(main())
