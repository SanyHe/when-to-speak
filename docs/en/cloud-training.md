# Cloud GPU training runbook (full-parameter SFT + GRPO)

**English** | [简体中文](../cloud-training.md)

> Training **never runs locally**; it always runs on a **cloud GPU in China**. This document uses **AutoDL** (a mainstream hourly GPU rental platform in China) as the example; Volcengine or Alibaba Cloud PAI work the same way.

## 1. Sizing

| Item | Recommendation |
|---|---|
| Platform | AutoDL (autodl.com) |
| GPU | RTX 4090 / 3090 (24GB) to start; A100 40GB is safer for GRPO |
| Image | PyTorch 2.x + CUDA 12.1 (platform image) |
| Training | **Full-parameter fine-tuning** (no LoRA, no 4-bit quantization, no Unsloth) |
| Serving | **vLLM** (OpenAI-compatible API), see §6 |
| VRAM budget | 0.6B full SFT ≈ 11–13GB; GRPO (G=8) ≈ 18–22GB |
| Workload | SFT 800 samples / 2 epochs; GRPO 300 prompts × 8 rollouts / 1 epoch |

VRAM breakdown (bf16 weights, fp32 AdamW states):

| Stage | Weights | Gradients | Optimizer | Reference model | Activations / rollout | Total (est.) |
|---|---|---|---|---|---|---|
| Full SFT | 1.2GB | 1.2GB | 4.8GB | – | 4–6GB | ≈ 11–13GB |
| GRPO | 1.2GB | 1.2GB | 4.8GB | 1.2GB (beta>0) | 6–10GB | ≈ 18–22GB |

> A 24GB card works if you lower `--batch_size` / `--num_generations`; on OOM drop to
> `--num_generations 4` first, then `--beta 0` (disable the reference model).

## 2. Initialise a fresh instance

```bash
# 1. Confirm the GPU
nvidia-smi

# 2. Clone the repo
cd /root
git clone <your-repo-url> when-to-speak
cd when-to-speak

# 3. Install training dependencies
pip install -r training/requirements.txt

# 4. Verify the toolchain (no GPU needed)
python3 -m pytest training/tests/ -q

# 5. Rebuild the full dataset (fixed seed, reproducible)
python3 training/src/build_dataset.py
```

Expected `dataset_stats.json`:

- SFT 800 / GRPO prompts 300 / Eval 200
- all three actions represented
- `near_duplicate_cross_split: 0`

> **One-shot run**: §3–§5 are wrapped by `./training/run_all.sh` (dataset → SFT → GRPO → eval).
> Override hyperparameters with environment variables, e.g. `SFT_BATCH=4 GRPO_GENERATIONS=4 ./training/run_all.sh`.
> When you need the online service afterwards, start vLLM per §6. The sections below are the equivalent
> step-by-step commands and troubleshooting notes; for a first run it is worth executing them one by one.

## 3. Full-parameter SFT

```bash
python3 training/src/full_sft.py \
  --model_path Qwen/Qwen3-0.6B \
  --data data/synthetic/sft_full.jsonl \
  --output_dir output/full_sft \
  --epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --learning_rate 2e-5 \
  --seed 42
```

`--model_path` accepts a Hugging Face repo id or a **local directory**. On Chinese networks, download the
weights to the instance first and point at the local path, so training does not refetch mid-run:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download Qwen/Qwen3-0.6B --local-dir /root/models/Qwen3-0.6B

python3 training/src/full_sft.py --model_path /root/models/Qwen3-0.6B ...
```

With the one-shot script, pass it as an environment variable: `BASE_MODEL=/root/models/Qwen3-0.6B ./training/run_all.sh`.

Outputs:

- `output/full_sft/final/` — full HF weights (~1.2GB, not committed)
- `output/full_sft/run_config.json` — hyperparameters and seed
- `output/full_sft/train_log.json` — loss curve

Implementation notes:

- Loss is computed only on assistant (JSON decision) tokens, with the prompt masked to -100.
- `enable_thinking=False`, so Qwen3 cannot emit thinking blocks that would corrupt the strict JSON target.
- Full-parameter training uses a low learning rate (2e-5); the prompt-masking logic is unit-tested (`test_full_sft.py`).

## 4. GRPO reinforcement learning

```bash
python3 training/src/grpo.py \
  --model_path output/full_sft/final \
  --data data/synthetic/grpo_prompts.jsonl \
  --output_dir output/grpo \
  --epochs 1 \
  --batch_size 2 \
  --num_generations 16 \
  --temperature 1.1 \
  --beta 0.04 \
  --seed 42
```

Reward functions (`training/src/rewards.py`, deterministic rules, no reward model):

| Function | Rule | Score |
|---|---|---|
| `format_reward` | strict schema pass / parseable JSON only / unparseable | +1.0 / +0.5 / 0 |
| `action_reward` | correct action / unnecessary notify / missed notify / digest↔silence | +1.0 / -0.5 / -1.0 / -0.15 |
| `channel_reward` | correct action and channel | +0.25 |
| `urgency_reward` | correct action and matching urgency | +0.10 |

> The asymmetric penalties (miss -1.0 > noise -0.5) are a product constraint: better one extra digest than a dropped P0.
>
> `num_generations 16` + `temperature 1.1` + `urgency_reward` respond to the first GRPO round's diagnosis:
> 66.7% of groups had identical rewards across 8 rollouts (no advantage available), and format and channel
> rewards were near saturation. More samples, a higher temperature and a finer reward term create within-group
> separation. If VRAM is short, drop back to `GRPO_GENERATIONS=8` first.
>
> TRL requires `generation_batch_size` to be divisible by `num_generations`; the script rounds
> `batch_size × grad_accum_steps` up automatically (8 → 16 when G=16), so `--grad_accum_steps` needs no manual change.

Outputs:

- `output/grpo/final/` — full HF weights after GRPO (the released model)
- `output/grpo/run_config.json` — hyperparameters, seed, reward function names, sample counts
- `output/grpo/train_log.json` — reward / loss / completion-length curves
- `output/grpo_v1/` — logs of the first GRPO round (G=8, temperature 1.0), kept for comparison;
  it tied with SFT on the 200-event eval, and the second round (G=16, temperature 1.1, urgency reward) is the released model

> The released model was first written to `output/grpo_v2/` and renamed to `output/grpo/` once finalised, with the
> first round moved to `output/grpo_v1/`. The `output_dir` fields inside both `run_config.json` files still record
> the original runtime paths: that is training fact and was not rewritten.

Implementation and version notes:

- Uses TRL `GRPOTrainer`; `trl>=0.19`. The tokenizer argument is named `tokenizer` in some versions and
  `processing_class` in others; the script picks whichever the installed signature accepts.
- `beta` is the KL coefficient against the SFT reference model; `beta 0` disables the reference model to save VRAM.
- GRPO depends on the full-SFT weights, so it must run after the previous step.
- If `GRPOConfig` fields differ from the installed version, check
  `python -c "import trl; help(trl.GRPOConfig)"` and record the deviation in `private/trae-progress.md`.

**Local evidence before spending GPU time**: the reward functions and prompt masking are deterministically unit-tested
(`training/tests/test_rewards.py`, `test_full_sft.py`), so a green `pytest` proves the scoring rules match the data
contract; training metrics are only backfilled after the GPU run.

## 5. Real-model evaluation

```bash
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=Qwen/Qwen3-0.6B,sft=output/full_sft/final,sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json
```

Acceptance direction:

- SFT JSON validity ≥ 95%
- Noise rate: monotonic decrease across base → SFT → SFT+GRPO
- Miss rate: SFT+GRPO does not rise relative to SFT

Measured values (200 held-out events):

| Model | JSON validity | Noise rate | Miss rate | Macro-F1 |
|---|---|---|---|---|
| base | 0.00% | 81.00% | 0.00% | 0.106 |
| SFT | 99.50% | 3.50% | 18.42% | 0.899 |
| SFT + GRPO | 99.50% | 3.50% | 5.26% | 0.924 |

## 6. Serving with vLLM

After training, start an OpenAI-compatible server so the local page can call the real model.

### 6.1 Install

```bash
pip install vllm
# A 0.6B model is undemanding, but Qwen3 needs a reasonably recent vLLM:
python3 -c "import vllm; print(vllm.__version__)"
```

### 6.2 Start the server

```bash
# Loads output/grpo/final by default, served name wts, port 6006
ALLOWED_ORIGINS="http://localhost:8765 http://127.0.0.1:8765" \
  ./training/src/serve_vllm.sh

# Or point at the SFT weights for comparison:
./training/src/serve_vllm.sh output/full_sft/final
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `6006` | AutoDL custom-service port |
| `SERVED_NAME` | `wts` | Model name typed into the page |
| `API_KEY` | empty | When set, requests must carry `Authorization: Bearer <key>` |
| `ALLOWED_ORIGINS` | `*` | CORS origins, space separated; relax for local debugging, tighten in public |
| `MAX_MODEL_LEN` | `2048` | Must be ≥ the 1024 used in training |
| `GPU_UTIL` | `0.6` | A 0.6B model is small; 0.6 is plenty |

Self-test inside the instance:

```bash
curl -s http://127.0.0.1:6006/v1/models
```

### 6.3 Expose it publicly (AutoDL)

1. AutoDL console → instance → **Custom service**, map port `6006`.
2. You get an address like `https://<region>-6006.autodl.pro`.
3. Enter that address in the page's "Service address" field (**without `/v1`**).

> AutoDL custom-service domains can change after an instance restarts or migrates; trust the console.

### 6.4 Connect the local page

```bash
# Serve the static site locally (http:// is required: file:// has a null Origin and is rejected by CORS)
python3 -m http.server 8765 --directory web
```

Open `http://localhost:8765`, then in the "Remote inference (vLLM)" panel:

1. Fill in the service address, model name (`wts` by default) and an API key if required.
2. Click **Test connection**; it should report the model list containing `wts`.
3. Click **Enable**; a "vLLM Live" tab appears in the model switcher.
4. Switch to it and click **Step** — the page sends the current event to vLLM and shows the returned JSON decision.

Notes:

- Requests use exactly the system prompt seen in training (`evaluate.py` writes it into
  `web/data/traces.json` as `_meta.system_prompt`) and explicitly disable Qwen3 thinking.
- Live results cache per event; requests happen only on tab switch or step, never in a background batch.
- "Live metrics" only count requested events; the rest stay pending.
- A browser-side call means the API key is visible to anyone who can open the page. Fine for local use;
  for a public deployment use a keyless internal port or add a backend proxy.

### 6.5 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to fetch` on test | CORS: `ALLOWED_ORIGINS` lacks `http://localhost:8765`; or the service address includes `/v1` |
| `HTTP 401` | Server has `API_KEY` set and the page did not supply it |
| `HTTP 404` | Model name differs from `SERVED_NAME` |
| Output prefixed with ` thinking` | vLLM too old for `chat_template_kwargs`; upgrade vLLM or let the frontend strip it |
| Out of memory | Lower `GPU_UTIL`, or set `EAGER=1` to disable CUDA graphs |

---

## 7. Sending evidence back

After training and evaluation, send the following back.

**Commit to the repo (small, public, reviewable)**

- `output/full_sft/run_config.json`, `output/full_sft/train_log.json`
- `output/grpo/run_config.json`, `output/grpo/train_log.json` (the reward curve is the only proof of GRPO convergence)
- `data/eval/eval_report_hf.json` / `eval_report_hf.md`, `data/eval/errors_hf.jsonl`
- `web/data/traces.json` (real traces produced by `--mode hf --web-out ...`)

**Not committed (too large or not publishable)**

- `output/**/final/` weights and `output/**/checkpoints/` optimizer states (publish via GitHub Release / HuggingFace)
- Raw terminal output of training (only a few final loss / reward lines belong in `private/` notes)

Also check the three-model traces in `web/data/traces.json` against the web stats, confirming switching and aggregation agree.

## 8. Downloading artifacts

```bash
# Weights: via the AutoDL console or scp; never in the code repo

# Committed configs and logs
scp -P <port> root@<host>:~/experiment/when-to-speak/output/full_sft/run_config.json output/full_sft/
scp -P <port> root@<host>:~/experiment/when-to-speak/output/full_sft/train_log.json  output/full_sft/
scp -P <port> root@<host>:~/experiment/when-to-speak/output/grpo/run_config.json  output/grpo/
scp -P <port> root@<host>:~/experiment/when-to-speak/output/grpo/train_log.json  output/grpo/
```

## 9. FAQ

- **OOM (SFT)**: lower `--batch_size`, raise `--grad_accum_steps`; `max_seq_length` defaults to 1024.
- **OOM (GRPO)**: drop `--num_generations` first (8 → 4), then set `--beta 0` to disable the reference model.
- **GRPO not converging / reward stuck at 0**: check that Qwen3 thinking is off (the script sets `enable_thinking=False`),
  then check that the gold field in `grpo_prompts.jsonl` is not dropped by `remove_unused_columns` (the script sets it to False).
- **TRL API mismatch**: run `python -c "import trl; print(trl.__version__)"`, adapt to the real signature and record the deviation in `private/trae-progress.md`.
- **Slow HuggingFace downloads**: set `HF_ENDPOINT=https://hf-mirror.com`, or pre-download the model and point `--model_path` at it.
