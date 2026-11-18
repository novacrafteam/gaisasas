"""Local Windows/Linux GUI for training (Tkinter + matplotlib). Start:  python train_gui.py

It launches train.py / dataset tools as separate processes and reads outputs/metrics.jsonl for live charts.
Stop is SAFE: it creates <output>/STOP; train.py then saves a checkpoint and exits. Resume continues from it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from chat_utils import gpu_stats, ram_stats, read_metrics
from config import PROJECT_DIR, default_output_dir

OUT = Path(default_output_dir())
PY = sys.executable
FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("GorphinAI Fine-Tuning")
        root.geometry("1100x820")
        self.proc: subprocess.Popen | None = None
        self.preset = tk.StringVar(value="local")
        self.vals = {k: tk.StringVar(value="-") for k in
                     ["Stage", "Epoch", "Step", "Train loss", "Val loss", "Learning rate", "Speed (s/step)", "VRAM", "RAM", "GPU temp"]}

        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        for i, k in enumerate(self.vals):
            ttk.Label(top, text=k + ":", font=("", 9, "bold")).grid(row=i // 5, column=(i % 5) * 2, sticky="e", padx=(8, 2))
            ttk.Label(top, textvariable=self.vals[k], width=16).grid(row=i // 5, column=(i % 5) * 2 + 1, sticky="w")
        self.bar = ttk.Progressbar(root, maximum=100)
        self.bar.pack(fill="x", padx=8)

        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(10, 4.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        row = ttk.Frame(root)
        row.pack(fill="x", padx=8)
        ttk.Label(row, text="Preset").pack(side="left")
        ttk.Combobox(row, textvariable=self.preset, values=["local", "kaggle", "cpu-test"], width=9, state="readonly").pack(side="left", padx=4)
        for text, cmd in [("Start Training", self.start), ("Stop Training", self.stop), ("Resume Training", self.resume),
                          ("Generate Dataset", self.gen_data), ("Validate Dataset", self.validate), ("Test Model", self.test_model),
                          ("Open Output Folder", self.open_out)]:
            ttk.Button(row, text=text, command=cmd).pack(side="left", padx=3, pady=4)

        self.log = tk.Text(root, height=9, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x", padx=8, pady=6)
        self.tick()

    # ---- process helpers
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def spawn(self, args: list[str]):
        OUT.mkdir(parents=True, exist_ok=True)
        logf = open(OUT / "gui_subprocess.log", "ab")
        return subprocess.Popen([PY, *args], cwd=PROJECT_DIR, stdout=logf, stderr=subprocess.STDOUT, **FLAGS)

    def start(self, resume=False):
        if self.running():
            messagebox.showinfo("Busy", "A process is already running.")
            return
        if not (PROJECT_DIR / "data" / "train.jsonl").exists():
            messagebox.showwarning("No data", "Generate the dataset first.")
            return
        if not resume and (OUT / "metrics.jsonl").exists() and not messagebox.askyesno(
                "Previous run", "A previous metrics file exists. Starting a new run clears the charts "
                "(checkpoints are kept; use Resume to continue). Start anyway?"):
            return
        if not resume:
            (OUT / "metrics.jsonl").unlink(missing_ok=True)
        self.proc = self.spawn(["train.py", "--preset", self.preset.get()] + (["--resume"] if resume else []))

    def resume(self):
        self.start(resume=True)

    def stop(self):
        if not self.running():
            return
        (OUT / "STOP").write_text("stop", encoding="utf-8")
        messagebox.showinfo("Stopping", "Stop requested: a checkpoint will be saved after the current step.")

    def _chain(self, cmds: list[list[str]]):
        def run():
            for c in cmds:
                self.proc = self.spawn(c)
                self.proc.wait()
        if self.running():
            messagebox.showinfo("Busy", "A process is already running.")
            return
        threading.Thread(target=run, daemon=True).start()

    def gen_data(self):
        self._chain([["dataset_generator.py", "--count", "600", "--out", "data/all.jsonl"],
                     ["dataset_validator.py", "data/all.jsonl"],
                     ["dataset_splitter.py", "data/all.jsonl"]])

    def validate(self):
        self._chain([["dataset_validator.py", "data/all.jsonl"]])

    def test_model(self):
        adapter = OUT / "adapter"
        args = ["chat_ui.py"] + (["--adapter", str(adapter)] if adapter.exists() else [])
        if self.preset.get() == "local":
            args.append("--qlora")
        if not adapter.exists() and not messagebox.askyesno("No adapter", "No trained adapter yet. Chat with the BASE model?"):
            return
        self.spawn(args)

    def open_out(self):
        OUT.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(OUT)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(OUT)])

    # ---- live update
    def tick(self):
        try:
            rows = read_metrics(OUT / "metrics.jsonl")
            logs = [r for r in rows if r.get("event") == "log"]
            stages = [r for r in rows if r.get("event") == "stage"]
            if stages:
                self.vals["Stage"].set(stages[-1]["stage"])
            if logs:
                last = logs[-1]
                self.vals["Epoch"].set(f"{last.get('epoch') or 0:.2f}")
                self.vals["Step"].set(f"{last['step']}/{last.get('max_steps', '?')}")
                if last.get("max_steps"):
                    self.bar["value"] = 100 * last["step"] / last["max_steps"]
                tl = [r for r in logs if r.get("loss") is not None]
                vl = [r for r in logs if r.get("eval_loss") is not None]
                if tl:
                    self.vals["Train loss"].set(f"{tl[-1]['loss']:.4f}")
                if vl:
                    self.vals["Val loss"].set(f"{vl[-1]['eval_loss']:.4f}")
                if last.get("learning_rate") is not None:
                    self.vals["Learning rate"].set(f"{last['learning_rate']:.2e}")
                if last.get("sec_per_step"):
                    self.vals["Speed (s/step)"].set(f"{last['sec_per_step']:.1f}")
            g, r_ = gpu_stats(), ram_stats()
            if g:
                self.vals["VRAM"].set(f"{g['vram_used_mb']/1024:.1f}/{g['vram_total_mb']/1024:.1f} GB")
                self.vals["GPU temp"].set(f"{g['gpu_temp_c']:.0f} °C")
            else:
                self.vals["VRAM"].set("n/a")
                self.vals["GPU temp"].set("n/a")
            if r_:
                self.vals["RAM"].set(f"{r_['ram_used_gb']}/{r_['ram_total_gb']} GB")
            from visualizer import plot_metrics

            plot_metrics(OUT / "metrics.jsonl", fig=self.fig)
            self.canvas.draw_idle()
            lf = OUT / "train.log"
            if lf.exists():
                tail = lf.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                self.log.configure(state="normal")
                self.log.delete("1.0", "end")
                self.log.insert("end", "\n".join(tail))
                self.log.configure(state="disabled")
        except Exception as e:  # noqa: BLE001  (never let the UI timer die)
            print("GUI update error:", e, file=sys.stderr)
        self.root.after(3000, self.tick)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
