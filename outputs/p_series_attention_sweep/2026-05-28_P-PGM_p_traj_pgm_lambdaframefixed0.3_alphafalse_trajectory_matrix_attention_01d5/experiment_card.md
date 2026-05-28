# Experiment Card

## Run identity

- Run name: 2026-05-28_P-PGM_p_traj_pgm_lambdaframefixed0.3_alphafalse_trajectory_matrix_attention_01d5
- Date: 2026-05-28T16:22:50
- Ablation ID: P-PGM
- Model variant: p_traj_pgm
- Dataset: diving48_v2
- Input type: precomputed_clip_vit_b16 frame embeddings
- Number of frames: 16
- Random seed: 0

## Architecture

Pipeline:
X [B, T, 512]
-> TemporalProjection
-> U [B, T, 128]
-> GaussianFramePGMSmoother(lambda_frame = 0.3)
-> Z [B, T, 128]
-> ProjectedPairwiseDiffNet
-> R [B, T-1, 128]
-> flatten R [B, 1920]
-> TrajectoryMatrixClassifier
-> logits [B, 48]

PGM interpretation:
If frame_pgm_smoother is enabled, frozen frame embeddings X_t are treated as noisy observations of latent clean frame states Z_t before DiffNet. If pgm_smoother is enabled after DiffNet, R_t is treated as noisy pairwise temporal evidence and smoothed into Y_t. The true Gaussian PGM information matrix is A = alpha I + lambda L, where L is the temporal path graph Laplacian. The learned InformationMatrixAccumulator is a learned sequential evidence accumulator, not the same object as A.

## Main purpose

Projected CLIP trajectory-matrix model with only fixed frame-level Gaussian PGM smoothing.

## Key config

| Field | Value |
|---|---|
| model.variant | p_traj_pgm |
| diff_nn.diff_net_type | pairwise_diff_net |
| diff_nn.hidden_dim | None |
| diff_nn.d_y | None |
| diff_nn.dropout | None |
| model.use_pre_pgm | False |
| pre_pgm.lambda_frame | None |
| model.use_pgm | True |
| pgm.lambda_frame | 0.3 |
| frame_pgm_smoother.type | none |
| frame_pgm_smoother.lambda_smooth | none |
| pgm_smoother.lambda_smooth | None |
| information_matrix.use_alpha | None |
| information_matrix.K | None |
| information_matrix.d_h | None |
| classifier.type | trajectory_matrix_attention |
| training.lr | 1e-4 |
| training.batch_size | 128 |
| training.epochs | 120 |
| training.early_stop_patience | 30 |

## Results

| Metric | Value |
|---|---|
| best_val_top1 | 0.18225190839694658 |
| best_val_epoch | 27 |
| epochs_trained | 57 |
| early_stopped | True |
| stop_reason | no validation improvement for 30 epochs |
| final_train_loss | 1.2608556668626645 |
| final_val_loss | 4.554447199552114 |
| final_train_top1 | 0.5971899893370131 |
| final_val_top1 | 0.15219465648854963 |
| test_top1 | null |

## Checkpoints

- best: C:\Users\ziyan\OneDrive\Desktop\PGM_project\outputs\p_series_attention_sweep\2026-05-28_P-PGM_p_traj_pgm_lambdaframefixed0.3_alphafalse_trajectory_matrix_attention_01d5\checkpoints\best.pt
- last: C:\Users\ziyan\OneDrive\Desktop\PGM_project\outputs\p_series_attention_sweep\2026-05-28_P-PGM_p_traj_pgm_lambdaframefixed0.3_alphafalse_trajectory_matrix_attention_01d5\checkpoints\last.pt

## Notes

- What worked:
- What failed:
- What to try next: Attention-head lambda sweep copied from P9_trajectory_matrix_linear_pgm_lam030.yaml. Same PGM placement/lambda as the source config; classifier is direct trajectory_matrix_attention over DiffNet relation tokens R. No InformationMatrixAccumulator.
