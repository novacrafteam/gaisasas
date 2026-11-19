import json, textwrap
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip().splitlines(True)})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(s).strip().splitlines(True)})

md("""
# GorphinAI Fine-Tuning on Kaggle
LoRA/QLoRA SFT of **Qwen/Qwen3.5-2B** (base weights are never modified).
**Before running:** Settings → Accelerator = *GPU T4 x2* (or T4), Internet = *On*. Run cells top to bottom.
Everything is stored in `/kaggle/working`. No Google Drive, Gradio or Docker is used.
""")
code('''
# ---- 1. SETTINGS (edit me) ----
REPO_URL     = "https://github.com/YOUR_USER/GorphinAI-FineTuning.git"   # <- your repository
BRANCH       = "main"
PRIVATE_REPO = False      # True -> a GitHub token is requested (hidden input) or read from Kaggle Secret GITHUB_TOKEN
DATASET_COUNT = 600       # examples to generate if the repo has no data/train.jsonl
NUM_EPOCHS    = 2
RESUME        = False      # True -> continue from the last checkpoint in /kaggle/working/outputs/checkpoints
RESTORE_FROM  = ""         # optional: path of a previous run's outputs dir attached as Kaggle input, e.g. /kaggle/input/my-run/outputs
EXPORT_MERGED = True       # True -> also merge adapter into base weights (.safetensors + .pt)
EXPORT_GGUF   = True       # True -> also convert to .gguf via llama.cpp (saved in /kaggle/working/gorphinai.gguf)
''')
code('''
# ---- 2. GPU / RAM check ----
import subprocess, shutil, os
try:
    import psutil
    vm = psutil.virtual_memory(); print(f"RAM: {vm.total/1e9:.1f} GB total, {vm.available/1e9:.1f} GB free")
except ImportError:
    print("psutil missing (will be installed below)")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("!! No GPU found. Enable it in Settings -> Accelerator, then restart the session.")
print("Disk free in /kaggle/working: %.1f GB" % (shutil.disk_usage("/kaggle/working").free/1e9) if os.path.exists("/kaggle/working") else "Not on Kaggle (paths will be local)")
''')
code('''
# ---- 3. Clone the project (token is never printed or saved) ----
import os, subprocess, getpass, base64, shutil, glob
BASE = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
os.chdir(BASE)
DEST = os.path.join(BASE, "repo")
token = ""
if PRIVATE_REPO:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        token = getpass.getpass("GitHub token (hidden, not stored): ")
if os.path.exists(DEST):
    shutil.rmtree(DEST)
cmd = ["git"]
if token:
    b64 = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    cmd += ["-c", f"http.extraHeader=Authorization: Basic {b64}"]   # transient: not written to .git/config
cmd += ["clone", "--depth", "1", "--branch", BRANCH, REPO_URL, DEST]
r = subprocess.run(cmd, capture_output=True, text=True)
msg = r.stdout + r.stderr
for secret in filter(None, [token, locals().get("b64")]):
    msg = msg.replace(secret, "***")
print(msg.strip())
if r.returncode != 0:
    raise SystemExit("git clone failed (check REPO_URL / BRANCH / token)")
del token
hits = glob.glob(os.path.join(DEST, "**", "train.py"), recursive=True)
if not hits:
    raise SystemExit("train.py not found in the cloned repository")
PROJECT = os.path.dirname(sorted(hits, key=len)[0])
os.chdir(PROJECT)
import sys; sys.path.insert(0, PROJECT)
print("Project directory:", PROJECT)
''')
code('''
# ---- 4. Install dependencies and check compatibility ----
import subprocess, sys
def pip(*a): return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], capture_output=True, text=True)
r = pip("-U", "transformers", "peft", "accelerate", "bitsandbytes", "psutil", "matplotlib", "huggingface_hub", "gguf")
print(r.stdout[-500:] + r.stderr[-1500:])
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)  # Kaggle's 0.10.0 breaks peft
# Optional speed-ups (compiling from source on Kaggle can take several minutes):
# r = pip("flash-linear-attention", "causal-conv1d")
# print("optional fast kernels:", "installed" if r.returncode == 0 else "not available (training still works, slower)")

import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| CUDA:", torch.cuda.is_available())
from model_loader import check_environment
for w in check_environment(): print("WARNING:", w)
assert torch.cuda.is_available(), "No GPU available"
''')
code('''
# ---- 5. Verify the model repo exists (weights are downloaded by train.py) ----
from huggingface_hub import model_info
info = model_info("Qwen/Qwen3.5-2B")
print("Found:", info.id, "| last modified:", info.last_modified)
''')
code('''
# ---- 6. Dataset: use the repo's data or generate it ----
import os, subprocess, sys
if not os.path.exists("data/train.jsonl") or not os.path.exists("data/validation.jsonl"):
    for c in (["dataset_generator.py", "--count", str(DATASET_COUNT), "--out", "data/all.jsonl"],
              ["dataset_validator.py", "data/all.jsonl"],
              ["dataset_splitter.py", "data/all.jsonl"]):
        print(subprocess.run([sys.executable, *c], capture_output=True, text=True).stdout[-800:])
else:
    print(subprocess.run([sys.executable, "dataset_validator.py", "data/train.jsonl"], capture_output=True, text=True).stdout[-600:])
''')
code('''
# ---- 7. (optional) restore a previous run's checkpoints to continue training ----
import shutil, os
OUT = "/kaggle/working/outputs"
if RESTORE_FROM:
    shutil.copytree(RESTORE_FROM, OUT, dirs_exist_ok=True)
    print("Restored", RESTORE_FROM, "->", OUT)
''')
code('''
# ---- 8. TRAIN with live progress (interrupt the cell = safe stop with checkpoint) ----
import subprocess, sys, time, os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from visualizer import plot_metrics
from chat_utils import read_metrics

OUT = "/kaggle/working/outputs"; os.makedirs(OUT, exist_ok=True)
METRICS, LOG, STOP = f"{OUT}/metrics.jsonl", f"{OUT}/stdout.log", f"{OUT}/STOP"
cmd = [sys.executable, "train.py", "--preset", "kaggle", "--num-train-epochs", str(NUM_EPOCHS)] + (["--resume"] if RESUME else [])
if not RESUME and os.path.exists(METRICS): os.remove(METRICS)

def render():
    rows = read_metrics(METRICS)
    logs = [r for r in rows if r.get("event") == "log"]; stages = [r for r in rows if r.get("event") == "stage"]
    clear_output(wait=True)
    if stages: print("Stage:", stages[-1]["stage"])
    if logs:
        l = logs[-1]; ms = l.get("max_steps") or 1
        pct = 100 * l["step"] / ms
        print(f"Step {l['step']}/{ms} ({pct:.0f}%) | epoch {l.get('epoch') or 0:.2f} | s/step {l.get('sec_per_step', float('nan')):.1f} | "
              f"VRAM {l.get('vram_used_mb', 0)/1024:.1f} GB | RAM {l.get('ram_used_gb', '?')} GB | GPU {l.get('gpu_temp_c', '?')} C")
        print("[" + "#" * int(pct // 4) + "." * (25 - int(pct // 4)) + "]")
    plot_metrics(METRICS, show=True); plt.show(); plt.close("all")
    if os.path.exists(LOG): print("".join(open(LOG, errors="replace").readlines()[-6:]))

proc = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
try:
    while proc.poll() is None:
        time.sleep(20); render()
except KeyboardInterrupt:
    open(STOP, "w").write("stop"); print("Stop requested - saving checkpoint ..."); proc.wait(timeout=900)
render(); print("train.py exit code:", proc.returncode, "(0 = ok, 3 = CUDA OOM: lower max_seq_length / batch size in config.py preset)")
''')
code('''
# ---- 9. Evaluate: BEFORE (base) vs AFTER (LoRA) ----
import subprocess, sys
r = subprocess.run([sys.executable, "evaluate.py", "--adapter", f"{OUT}/adapter"], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1500:] if r.returncode else "")
''')
code('''
# ---- 10. Interactive test: type ANY prompt (adapter must exist) ----
from chat_ui import notebook_ui
session = notebook_ui(f"{OUT}/adapter")
''')
code('''
# ---- 11. Export (.safetensors + .pt + .gguf) and archive for download ----
import subprocess, sys, shutil
cmd = [sys.executable, "export_model.py", "--adapter", f"{OUT}/adapter", "--zip", "--pt"]
if EXPORT_MERGED or EXPORT_GGUF:
    cmd += ["--merge", f"{OUT}/merged"]
if EXPORT_GGUF:
    cmd += ["--gguf", "/kaggle/working/gorphinai.gguf", "--fetch-llama-cpp"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2500:]); print(r.stderr[-1500:] if r.returncode else "")
shutil.copy2(f"{OUT}/gorphinai_adapter.zip", "/kaggle/working/gorphinai_adapter.zip")
print("Ready in /kaggle/working (Output tab): gorphinai_adapter.zip" + (" + gorphinai.gguf" if EXPORT_GGUF else ""))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
Path(r"i:\GorphinAI\GorphinAI-FineTuning\kaggle_gorphinai.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("cells:", len(cells))
