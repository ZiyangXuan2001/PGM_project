# Embedding Difference PGM Model

This project implements a controlled PyTorch scaffold for embedding-level
temporal reasoning on Diving48 V2.

The current implemented backbone path uses precomputed OpenAI CLIP ViT-B/16
frame embeddings:

```text
X: [B, T, 512]
```

The model predicts 48 Diving48 V2 classes.

## Model Idea

The full model pipeline is:

1. `PairwiseDiffNet` computes adjacent-frame difference embeddings `R_t`.
2. `GaussianPGMSmoother` smooths `R_t` into latent trajectory states `Y_t`.
3. `InformationMatrixAccumulator` summarizes variable-length `Y` into fixed-size `H_final`.
4. A classifier predicts class logits.

Raw frame encoding and temporal reasoning are separated. The first controlled
training version assumes CLIP frame embeddings have already been saved.

## Controlled Options

The code intentionally supports only a small set of options:

- backbone: `precomputed_clip_vit_b16`
- DiffNN: `pairwise_diff_net`
- PGM smoother: `none`, `gaussian_chain`, `learnable_gaussian_chain`
- information matrix: `accumulator` with `use_alpha: true/false`
- classifier: `mlp`, `attention_pool`

Controlled ablation variants:

| ID | Variant | Architecture |
|---|---|---|
| E0 | `mean_pool_baseline` | `X -> mean_pool(X) -> MLP` |
| E1 | `diff_only` | `X -> PairwiseDiffNet -> R -> mean_pool(R) -> MLP` |
| E2 | `diff_pgm` | `X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> mean_pool(Y) -> MLP` |
| E3 | `diff_pgm_info` | `X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> InformationMatrixAccumulator -> H -> MLP` |
| E4 | `diff_pgm_info_attention` | `X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> InformationMatrixAccumulator -> H -> AttentionHead` |

The simple baseline path is:

```text
X [B, T, 512] -> mean_pool(X) -> MLP -> logits [B, 48]
```

## Optional CLIP Backbone

`CLIPFrameEncoder` uses OpenAI CLIP `ViT-B/16`, matching AIM's use of CLIP-pretrained ViT-B/16 and ViT-L/14 image backbones. It encodes each frame independently and performs no temporal mixing.

CLIP is optional and not included in `requirements.txt`. To install it:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Setup

```bash
pip install -r requirements.txt
```

CLIP embedding extraction also needs OpenAI CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Smoke Tests

Run the controlled shape tests:

```bash
python scripts/smoke_test.py
```

Use a custom config path:

```bash
python scripts/smoke_test.py --config configs/default.yaml
```

Run unittest smoke coverage:

```bash
python -m unittest tests/test_controlled_model_shapes.py
```

The smoke tests use `X = torch.randn(2, 16, 512)` and verify:

- `R: [B, T-1, d_y]`
- `Y: [B, T-1, d_y]`
- `H_final: [B, K, d_h]`
- `logits: [B, 48]`

Minimal usage:

```python
import torch
from models import EmbeddingDifferencePGMModel

model = EmbeddingDifferencePGMModel.from_config(config)
X = torch.randn(2, 16, 512)
logits = model(X)
print(logits.shape)  # [2, 48]
```

## Diving48 V2 Dataset

The project is configured for Diving48 V2 by default. Put the dataset on the
training machine in this layout:

```text
data/diving48_v2/
  annotations/
    Diving48_V2_train.json
    Diving48_V2_test.json
    Diving48_vocab.json
  videos/
    <vid_name>.mp4
```

If you have pre-extracted RGB frames instead of mp4 files, use:

```text
data/diving48_v2/
  rawframes/
    <vid_name>/
      img_00001.jpg
      img_00002.jpg
```

Check the local dataset without downloading anything:

```bash
python scripts/check_diving48_dataset.py \
  --dataset_root data/diving48_v2 \
  --input_format auto
```

Extract CLIP frame embeddings:

```bash
python scripts/extract_diving48_clip_embeddings.py \
  --dataset_root data/diving48_v2 \
  --input_format auto \
  --num_frames 16 \
  --backbone_name ViT-B/16 \
  --embedding_subdir clip_vit_b16 \
  --device cuda
```

This writes:

```text
data/diving48_v2/embeddings/clip_vit_b16/train.pt
data/diving48_v2/embeddings/clip_vit_b16/test.pt
```

Diving48 V2 ships train/test annotations. For the first controlled version,
pass `test.pt` as the validation file unless you create a held-out validation
split. If you want that held-out split, add `--val_fraction 0.1` during
embedding extraction and train with `val.pt`.

Check saved embeddings:

```bash
python scripts/check_diving48_embeddings.py \
  --emb_dir data/diving48_v2/embeddings/clip_vit_b16
```

Train the default model on Diving48 V2 embeddings:

```bash
python scripts/train_embeddings.py \
  --config configs/default.yaml \
  --train_file data/diving48_v2/embeddings/clip_vit_b16/train.pt \
  --val_file data/diving48_v2/embeddings/clip_vit_b16/test.pt \
  --device cuda
```

Useful controlled overrides:

```bash
python scripts/train_embeddings.py \
  --config configs/default.yaml \
  --train_file data/diving48_v2/embeddings/clip_vit_b16/train.pt \
  --val_file data/diving48_v2/embeddings/clip_vit_b16/test.pt \
  --ablation_id E3 \
  --lambda_smooth 1.0 \
  --use_alpha true \
  --device cuda
```

Every training run writes a local run folder containing:

```text
outputs/runs/<run_name>/
  config_resolved.yaml
  metrics.json
  train_log.csv
  model_summary.txt
  experiment_card.md
  checkpoints/best.pt
  checkpoints/last.pt
```

Completed runs are also appended to:

```text
experiments/experiment_registry.csv
```

Inspect local runs:

```bash
python scripts/list_experiments.py --sort best_val_top1 --last 10
```

## Warm-up Dataset

The `warmup_motion_manim` dataset is a small Manim-generated video dataset for
testing whether temporal modeling helps before moving to a final dataset. Each
sample is a short clean 2D geometric motion sequence, and the class depends on
the trajectory over time rather than object color, shape, size, background, or
initial position.

Manim Community Edition is required for generation:

```bash
pip install manim
```

Classes:

- `0`: `clockwise`
- `1`: `counter_clockwise`
- `2`: `horizontal_oscillation`
- `3`: `vertical_oscillation`
- `4`: `stationary`

Generate raw videos:

```bash
python scripts/generate_warmup_motion_manim.py \
  --out_dir data/warmup_motion_manim/raw \
  --num_train 1000 \
  --num_val 200 \
  --num_test 200 \
  --T 16 \
  --image_size 224 \
  --seed 0 \
  --preview true
```

Check the raw dataset:

```bash
python scripts/check_warmup_dataset.py
```

Extract CLIP frame embeddings:

```bash
python scripts/extract_clip_embeddings.py \
  --backbone_name ViT-B/16 \
  --embedding_subdir clip_vit_b16 \
  --device mps
```

Check saved embeddings:

```bash
python scripts/check_warmup_embeddings.py \
  --emb_dir data/warmup_motion_manim/embeddings/clip_vit_b16
```

The raw videos are only used to generate embeddings. The temporal model trains
on saved tensors `X: [N, T, 512]` and labels from:

```text
data/warmup_motion_manim/embeddings/clip_vit_b16/train.pt
data/warmup_motion_manim/embeddings/clip_vit_b16/val.pt
data/warmup_motion_manim/embeddings/clip_vit_b16/test.pt
```

For the experiment grid, generate both embedding sets:

```bash
python scripts/extract_clip_embeddings.py \
  --backbone_name ViT-B/32 \
  --embedding_subdir clip_vit_b32 \
  --device mps

python scripts/extract_clip_embeddings.py \
  --backbone_name ViT-B/16 \
  --embedding_subdir clip_vit_b16 \
  --device mps
```

## Experiment Grid

The controlled Diving48 V2 grid is configured in `configs/experiment_grid.yaml`.
It trains only on precomputed embeddings; CLIP is not trained inside the
training loop.

Main full grid:

- `pgm_smoother.type`: `{none, gaussian_chain}`
- `classifier.type`: `{mlp, attention_pool}`
- `lambda_smooth`: `{0.3, 1.0, 2.0}`
- `use_alpha`: `{false, true}`
- total runs: `2 x 2 x 3 x 2 = 24`

When `pgm_smoother.type=none`, the model uses `Y = R`. When
`pgm_smoother.type=gaussian_chain`, it solves `(I + lambda_smooth * L_path)Y = R`.

Run the debug grid:

```bash
python scripts/run_experiment_grid.py --stage debug --device cuda
```

Run the full grid:

```bash
python scripts/run_experiment_grid.py --stage full --device cuda
```

Summarize results:

```bash
python scripts/summarize_grid_results.py
```
