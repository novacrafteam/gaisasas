"""Training visualisation from <output_dir>/metrics.jsonl (written by train.py).

  python visualizer.py                      # saves outputs/training_curves.png
  python visualizer.py --metrics path --show
In notebooks use: from visualizer import plot_metrics; plot_metrics(path)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chat_utils import read_metrics
from config import default_output_dir


def plot_metrics(metrics_path, save_to=None, show=False, fig=None):
    import matplotlib

    if not show and fig is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_metrics(metrics_path) if r.get("event") == "log"]
    if fig is None:
        fig = plt.figure(figsize=(12, 7))
    fig.clf()
    axes = fig.subplots(2, 2)
    if not rows:
        axes[0][0].text(0.1, 0.5, "No metrics yet", transform=axes[0][0].transAxes)
        return fig

    def series(key):
        pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    ax = axes[0][0]
    for key, label in (("loss", "train loss"), ("eval_loss", "validation loss")):
        x, y = series(key)
        if x:
            ax.plot(x, y, marker="o" if key == "eval_loss" else None, label=label)
    ax.set_title("Loss")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)
    x, y = series("learning_rate")
    axes[0][1].plot(x, y)
    axes[0][1].set_title("Learning rate")
    axes[0][1].grid(alpha=0.3)
    x, y = series("vram_used_mb")
    axes[1][0].plot(x, [v / 1024 for v in y])
    axes[1][0].set_title("GPU memory used (GB)")
    axes[1][0].grid(alpha=0.3)
    x, y = series("sec_per_step")
    axes[1][1].plot(x, y)
    axes[1][1].set_title("Seconds per optimizer step")
    axes[1][1].grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=str(Path(default_output_dir()) / "metrics.jsonl"))
    ap.add_argument("--out", default=str(Path(default_output_dir()) / "training_curves.png"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.metrics).exists():
        print(f"Metrics file not found: {args.metrics}. Train first.", file=sys.stderr)
        return 2
    fig = plot_metrics(args.metrics, args.out, args.show)
    print("Saved", args.out)
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
