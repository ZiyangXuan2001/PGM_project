# Experiment Card

## Run identity

- Run name:
- Date:
- Ablation ID:
- Model variant:
- Dataset:
- Input type:
- Number of frames:
- Random seed:

## Architecture

Pipeline:
X [B, T, 512]
-> ...
-> logits [B, 48]

## Main purpose

Write one or two sentences explaining what this run tests.

## Key config

| Field | Value |
|---|---|
| model.variant | |
| pgm_smoother.lambda_smooth | |
| information_matrix.use_alpha | |
| information_matrix.K | |
| information_matrix.d_h | |
| classifier.type | |
| training.lr | |
| training.batch_size | |
| training.epochs | |

## Results

| Metric | Value |
|---|---|
| best_val_top1 | |
| best_val_epoch | |
| final_train_loss | |
| final_val_loss | |
| final_train_top1 | |
| final_val_top1 | |
| test_top1 | |

## Checkpoints

- best:
- last:

## Notes

- What worked:
- What failed:
- What to try next:

