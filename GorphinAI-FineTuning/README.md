# GorphinAI Fine-Tuning

LoRA / QLoRA supervised fine-tuning of **`Qwen/Qwen3.5-2B`** into a calm, friendly, honest trilingual
(🇺🇦 / 🇷🇺 / 🇬🇧) assistant. The base weights are downloaded from Hugging Face and **never modified** – only a small
LoRA adapter is trained.

## Facts verified against the real model repo (not guessed)
- `Qwen/Qwen3.5-2B` exists, Apache-2.0, not gated. Architecture `Qwen3_5ForConditionalGeneration`
  (`model_type: qwen3_5`, 24 layers, hybrid: 3 × *linear-attention* + 1 × *full-attention*, with a vision tower and an MTP head).
- It needs a **recent `transformers`** (the model card says latest/main).
- Real linear-layer names (from `model.safetensors.index.json`):
  `self_attn.{q,k,v,o}_proj` (full-attention layers), `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (linear-attention layers),
  `mlp.{gate,up,down}_proj`. LoRA targets exactly these, **only inside `language_model`** (regex in `config.py`), so the vision tower and MTP head are untouched.
- Qwen3.5 defaults to "thinking" mode. GorphinAI is trained in **non-thinking** mode (`enable_thinking=False`);
  training and inference share one prompt builder (`chat_utils.build_prompt`).

## Layout
| File | Purpose |
|---|---|
| `config.py` | all settings + presets (`local`, `kaggle`, `cpu-test`), system prompts |
| `model_loader.py` | loads original weights (+4-bit for QLoRA), attaches LoRA, memory estimate |
| `dataset_generator.py` / `data_banks.py` | template + optional AI-assisted trilingual dataset |
| `dataset_validator.py`, `dataset_splitter.py` | quality checks, stratified train/val split |
| `train.py` | SFT (checkpoints, resume, safe stop, OOM handling, metrics log) |
| `evaluate.py` | base-vs-LoRA quality comparison (automatic checks + manual review files) |
| `chat_ui.py` | interactive tester with sliders, multi-turn, 👍/👎 → `feedback.jsonl` |
| `benchmark.py` | real speed / VRAM measurements |
| `visualizer.py`, `train_gui.py` | charts; local Tkinter control panel |
| `export_model.py` | zip adapter, optional merge |
| `kaggle_gorphinai.ipynb` | Kaggle notebook |
| `data/train.jsonl`, `data/validation.jsonl` | shipped starter dataset (synthetic) |

## 1. Dataset
```bash
python dataset_generator.py --count 600 --out data/all.jsonl   # ~33% uk / ru / en
python dataset_validator.py data/all.jsonl                      # report (add --fix --out clean.jsonl to clean)
python dataset_splitter.py data/all.jsonl --val-ratio 0.1
```
- **Template mode** works offline. Math answers and logic puzzles are *computed* (always correct); knowledge answers are
  hand-written. Unique capacity is limited (~500 examples with the default `--max-share 0.15`); the generator stops and
  tells you instead of duplicating. Extend `data_banks.py` for more.
- **AI-assisted mode** (optional, you provide access – nothing is bundled):
  `set GORPHIN_API_KEY=...` then `python dataset_generator.py --mode ai --api-base https://your-endpoint/v1 --model NAME --count 2000`.
  Any OpenAI-compatible `/chat/completions` endpoint works. Always run the validator and **spot-check by hand**.
- `--resume` keeps the existing file and only generates the missing records.
- All data is synthetic (`source` field) – it is not real user conversation.

> ⚠️ The starter data teaches **style, format, language and honesty**, not knowledge. 500 examples is small; for a
> production-quality assistant extend it (AI-assisted mode + manual review) and compare with `evaluate.py`.

## 2. Train locally (Windows 10/11, GTX 1660 Ti, 32 GB RAM)
```powershell
py -m venv .venv ; .\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124     # pick the CUDA build from pytorch.org
pip install -r requirements.txt
python train.py --preset local --dry-run      # checks data, config, prints a ROUGH VRAM estimate
python train.py --preset local                # or: python train_gui.py
```
`local` = QLoRA 4-bit, seq 512, batch 1 × accumulation 16, LoRA r=8, fp16 (the 1660 Ti has no bf16).
Notes: no guarantee against OOM – on OOM the script says what to lower (`--max-seq-length 256`, `--lora-r 4`).
Without `flash-linear-attention` the Qwen3.5 linear layers use a slow PyTorch fallback (on Windows those kernels are
usually unavailable), so expect slow steps; run `python benchmark.py --train-step --preset local` to measure it for real.
bitsandbytes needs a recent version with Windows support; if QLoRA import fails use `--no-qlora` only on GPUs with enough VRAM (the 2B model in fp16 is ≈ 4.5 GB before activations).

GUI: `python train_gui.py` – live charts/stats, Start / Stop / Resume, dataset buttons, Test Model, Open Output Folder.

## 3. Train on Kaggle
1. Push this folder to GitHub (the project can be the repo root or a sub-folder).
2. New Kaggle notebook → *File → Import notebook* → `kaggle_gorphinai.ipynb`.
3. Settings: **Accelerator: GPU T4**, **Internet: On**.
4. Edit the first cell (`REPO_URL`, `BRANCH`). Private repo: set `PRIVATE_REPO=True`; the token is read from the Kaggle
   Secret `GITHUB_TOKEN` or a hidden prompt; it is passed to git transiently, scrubbed from logs and never stored.
5. Run all cells. Interrupting the training cell triggers a safe stop with a checkpoint.
6. The last cell creates `/kaggle/working/gorphinai_adapter.zip` – download it from the *Output* tab.

**Continue training after the session ended:** *Save Version* → in a new notebook add that version's output as Input, set
`RESTORE_FROM="/kaggle/input/<name>/outputs"` and `RESUME=True`.

## 4. Test the model with your own prompts
```bash
python chat_ui.py --adapter outputs/adapter [--qlora]      # Tkinter: language, temperature, top-p, max tokens, repetition penalty, multi-turn
```
In Kaggle the notebook shows the same controls via ipywidgets. 👍/👎 saves the whole dialogue to
`outputs/feedback.jsonl` with `"reviewed": false`. **It is never used for training automatically.**

## 5. Evaluate (before vs after)
```bash
python evaluate.py --adapter outputs/adapter [--qlora]
```
Runs 24 fixed tests (uk/ru/en, code, math, logic, instruction following, multi-turn) on the base model
(adapter disabled) and on the tuned model with greedy decoding, prints a table, and writes `outputs/eval/review.md`
(side-by-side answers) and `human_scores.json` (fill in 1–5, then `python evaluate.py --human-summary outputs/eval/human_scores.json`).
Automatic checks are coarse; 24 tests means ±1 test is noise. Validation loss alone is not evidence of a better assistant.

## 6. Continue / export
```bash
python train.py --preset local --resume                 # continues from outputs/checkpoints/checkpoint-N
python export_model.py --adapter outputs/adapter --zip  # outputs/gorphinai_adapter.zip
python export_model.py --adapter outputs/adapter --merge outputs/merged   # optional full model (needs RAM)
```
Server deployment: install a recent `transformers` + `peft`, load the base `Qwen/Qwen3.5-2B`, then
`PeftModel.from_pretrained(base, "adapter_dir")` – or serve the merged folder with vLLM / SGLang / transformers.
Always build prompts with `enable_thinking=False`, as in `chat_utils.build_prompt`.
Merging should be done on the 16-bit model, not on a 4-bit one.

## Known limitations / honest status
- **Not tested on a GPU or with real weights in this environment**: the pieces verified were dataset generation/validation/splitting,
  Python syntax of all files, notebook validity, config/arg handling, and (on CPU) tokenizer/prompt/LoRA-target logic where noted in the final report.
  First real runs on Kaggle/GPU may still surface version-specific issues – the scripts print explicit errors.
- Starter dataset is small and templated; the model may sound repetitive until you add more diverse data.
- Loss is computed only on the final assistant message of each example.
- Qwen3.5 linear-attention speed depends on optional kernels (`flash-linear-attention`, `causal-conv1d`).
- The loader tries `AutoModelForCausalLM`, then `AutoModelForImageTextToText`; the hub checkpoint is a conditional-generation model, and
  which class loads cleanly depends on your transformers version.
- QLoRA on Windows depends on a bitsandbytes build that supports your CUDA setup.
- Your hardware note says 8 GB VRAM; a GTX 1660 Ti normally has 6 GB – scripts read the real value from the driver.
- Memory estimates are rough heuristics, labelled as such.
