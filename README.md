# CPU LoRA LLM Fine-Tuning Repo

Small, notebook-free repo for CPU-only LoRA fine-tuning of a Hugging Face causal language model.

Default model: `Qwen/Qwen3-1.7B`.

The repo intentionally separates three data groups:

- `data/manual/manual.jsonl`: examples you write by hand. By default these are added only to training.
- `data/augmented/augmented.jsonl`: examples already augmented elsewhere. This repo splits this file into train/val/test.
- `data/hard_test/hard_test.jsonl`: hard edge cases used only after training for final metrics.

The AI augmentation/generation step is intentionally **not** included.

## Data format

Each row is JSONL:

```json
{"id":"ex-001","instruction":"Classify sentiment.","input":"I love it.","output":"positive"}
```

`input` can be empty. Extra metadata fields are allowed and ignored by training.

## Quick start with Docker

Build:

```bash
docker build -t llm-finetune-cpu .
```

Train and evaluate:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/reports:/app/reports" \
  llm-finetune-cpu train --config configs/default.yaml
```

Evaluate an existing adapter:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/reports:/app/reports" \
  llm-finetune-cpu evaluate --config configs/default.yaml --adapter outputs/final_adapter
```


## Environment variables

Create or edit `.env` to cap data during workflow tests:

```bash
cp .env.example .env
# Then edit MAX_EXAMPLES. Empty or 0 means use all examples.
MAX_EXAMPLES=3
```

`MAX_EXAMPLES` limits each raw JSONL source before splitting: manual, augmented, and hard-test. Use `3` or more if you want the augmented split to keep train/val/test non-empty.

## Local run without Docker

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m llm_finetune train --config configs/default.yaml
```

## Outputs

Training writes:

- `outputs/checkpoints/`: step checkpoints
- `outputs/final_adapter/`: trained LoRA adapter
- `outputs/metrics/eval_summary.json`: baseline vs final metrics
- `outputs/metrics/log_history.json`: Trainer log history
- `reports/plots/training_loss.png`
- `reports/plots/eval_loss.png`
- `reports/plots/loss_improvement.png`
- `reports/plots/token_accuracy_improvement.png`

## Configuration

Edit `configs/default.yaml` to change:

- base model
- LoRA rank/alpha/dropout/target modules
- split ratios
- logging/evaluation frequency
- max length
- batch size and gradient accumulation

CPU fine-tuning is slow. Keep `max_length`, batch size, and epoch count conservative.

## Smoke test

For fast local validation without downloading the real model, use the tiny config:

```bash
python -m llm_finetune train --config configs/smoke.yaml
```

That config is for code-path testing only, not quality.
