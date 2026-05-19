# RunPod Guide

This project trains on precomputed CLIP ViT-B/16 frame embeddings for Diving48 V2.
RunPod should only run the existing E0-E4 controlled variants. It should not
download Diving48 or extract new backbones during these launch scripts.

## Recommended RunPod Setup

- Use a PyTorch GPU template with CUDA already installed.
- Store the repo under `/workspace/<repo_name>`.
- Store data under `/workspace/data/`.
- Keep outputs under `/workspace/<repo_name>/outputs/`.
- Enable SSH terminal access. Jupyter is optional.
- Do not store important results outside `/workspace`, because other container
  locations may be temporary.
- Stop or terminate the pod after use.

## Setup

```bash
cd /workspace
git clone <repo_url>
cd <repo_name>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the RunPod template already includes a CUDA-enabled PyTorch, keep that
version. If you install PyTorch yourself, use the CUDA wheel recommended by
the PyTorch website for your pod image.

## Environment Check

Run this before training:

```bash
python scripts/check_runpod_environment.py --config configs/default.yaml
```

It prints Python, PyTorch, CUDA, GPU, `/workspace`, config loading, output
writability, and random tensor forward checks for E0-E4.

## One-Command Main Interface

Use this first. It checks the environment, runs a tiny fake training job, and
prints `RUNPOD_MAIN_STATUS: OK` if the pipeline is healthy.

```bash
python scripts/runpod_main.py \
  --stage fake_small \
  --max-train-samples 128 \
  --epochs 2 \
  --batch-size 16 \
  --variants all
```

To download/prepare Diving48 V2, extract a tiny CLIP embedding subset, and run
a real tiny training job:

```bash
python scripts/runpod_main.py \
  --stage dataset_small \
  --dataset-root /workspace/data/diving48_v2 \
  --embeddings-root /workspace/data/diving48_embeddings \
  --download-videos \
  --max-extract-samples 16 \
  --max-train-samples 64 \
  --epochs 2 \
  --batch-size 16 \
  --variants E0,E4
```

Important: `--download-videos` downloads the full Diving48 video archive, which
is large. If the dataset is already present, use:

```bash
python scripts/runpod_main.py \
  --stage dataset_small \
  --dataset-root /workspace/data/diving48_v2 \
  --embeddings-root /workspace/data/diving48_embeddings \
  --skip-download \
  --max-extract-samples 16 \
  --epochs 2 \
  --variants E0,E4
```

If the official Diving48 URLs change, override them with `--train-url`,
`--test-url`, `--vocab-url`, and `--video-url`.

Note: the UCSD annotation URLs can return HTTP 403 from cloud machines. If that
happens, your GPU/model environment is still fine. Put the three annotation
files under `/workspace/data/diving48_v2/annotations/` manually, or use
OpenDataLab/MMAction2:

```bash
pip install -U openmim opendatalab
odl login
mim download mmaction2 --dataset diving48
```

The video archive default in this project uses the Hugging Face mirror:
`bkprocovid19/diving48`.

## Small Training Workflow

Fake-data training checks code, logging, checkpoints, and the experiment
registry. Accuracy is not meaningful.

```bash
python scripts/runpod_small_start.py \
  --mode fake \
  --max-samples 128 \
  --epochs 2 \
  --batch-size 16 \
  --variants all
```

Real tiny training checks that your local embedding file can be loaded and
trained on.

```bash
python scripts/runpod_small_start.py \
  --mode real \
  --embeddings-path /workspace/data/diving48_embeddings/train_embeddings.pt \
  --samples-per-class 2 \
  --epochs 3 \
  --batch-size 16 \
  --variants all
```

Overfit debugging checks whether the model can learn a tiny subset.

```bash
python scripts/runpod_small_start.py \
  --mode real \
  --embeddings-path /workspace/data/diving48_embeddings/train_embeddings.pt \
  --max-samples 64 \
  --overfit \
  --epochs 20 \
  --batch-size 16 \
  --variants E2,E3,E4
```

## Full Training

Expected embedding directory examples:

```text
/workspace/data/diving48_embeddings/
  train_embeddings.pt
  val_embeddings.pt
```

The full launcher also accepts `train.pt` and `val.pt`. If no validation file
exists, create one before full training. The script will not download data.

Check only:

```bash
python scripts/runpod_full_start.py \
  --stage check \
  --config configs/default.yaml
```

Train one variant:

```bash
python scripts/runpod_full_start.py \
  --stage train \
  --config configs/default.yaml \
  --embeddings-dir /workspace/data/diving48_embeddings \
  --variant diff_pgm_info_attention \
  --epochs 30 \
  --batch-size 64
```

Run full E0-E4 ablation:

```bash
python scripts/runpod_full_start.py \
  --stage ablation \
  --config configs/default.yaml \
  --embeddings-dir /workspace/data/diving48_embeddings \
  --variants all \
  --epochs 30 \
  --batch-size 64
```

Run check and ablation in one command:

```bash
python scripts/runpod_full_start.py \
  --stage all \
  --config configs/default.yaml \
  --embeddings-dir /workspace/data/diving48_embeddings \
  --variants all \
  --epochs 30 \
  --batch-size 64
```

## Outputs

Every run writes:

```text
outputs/runs/<run_name>/
  config_resolved.yaml
  train_log.csv
  metrics.json
  model_summary.txt
  experiment_card.md
  checkpoints/best.pt
  checkpoints/last.pt
```

The global registry is:

```text
experiments/experiment_registry.csv
```

Inspect recent runs:

```bash
python scripts/list_experiments.py --sort best_val_top1 --last 10
```

Download these after training:

- `outputs/runs/<run_name>/metrics.json`
- `outputs/runs/<run_name>/train_log.csv`
- `outputs/runs/<run_name>/experiment_card.md`
- `outputs/runs/<run_name>/checkpoints/best.pt`
- `experiments/experiment_registry.csv`

For a full archive, download:

```text
outputs/
experiments/experiment_registry.csv
```
