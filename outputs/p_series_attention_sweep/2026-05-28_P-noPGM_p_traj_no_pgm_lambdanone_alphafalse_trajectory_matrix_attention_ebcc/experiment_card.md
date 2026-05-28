# Experiment Card

## Run identity

- Run name: 2026-05-28_P-noPGM_p_traj_no_pgm_lambdanone_alphafalse_trajectory_matrix_attention_ebcc
- Date: 2026-05-28T16:15:30
- Ablation ID: P-noPGM
- Model variant: p_traj_no_pgm
- Dataset: diving48_v2
- Input type: precomputed_clip_vit_b16 frame embeddings
- Number of frames: 16
- Random seed: 0

## Architecture

Pipeline:
X [B, T, 512]
-> TemporalProjection
-> U [B, T, 128]
-> ProjectedPairwiseDiffNet
-> R [B, T-1, 128]
-> flatten R [B, 1920]
-> TrajectoryMatrixClassifier
-> logits [B, 48]

PGM interpretation:
If frame_pgm_smoother is enabled, frozen frame embeddings X_t are treated as noisy observations of latent clean frame states Z_t before DiffNet. If pgm_smoother is enabled after DiffNet, R_t is treated as noisy pairwise temporal evidence and smoothed into Y_t. The true Gaussian PGM information matrix is A = alpha I + lambda L, where L is the temporal path graph Laplacian. The learned InformationMatrixAccumulator is a learned sequential evidence accumulator, not the same object as A.

## Main purpose

Projected CLIP trajectory-matrix baseline with no frame PGM.

## Key config

| Field | Value |
|---|---|
| model.variant | p_traj_no_pgm |
| diff_nn.diff_net_type | pairwise_diff_net |
| diff_nn.hidden_dim | None |
| diff_nn.d_y | None |
| diff_nn.dropout | None |
| model.use_pre_pgm | None |
| pre_pgm.lambda_frame | None |
| model.use_pgm | False |
| pgm.lambda_frame | 0.0 |
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
| best_val_top1 | 0.1760496183206107 |
| best_val_epoch | 22 |
| epochs_trained | 52 |
| early_stopped | True |
| stop_reason | no validation improvement for 30 epochs |
| final_train_loss | 1.2326548582695662 |
| final_val_loss | 4.5808788656278425 |
| final_train_top1 | 0.5988835225490811 |
| final_val_top1 | 0.15458015267175573 |
| test_top1 | null |

## Checkpoints

- best: C:\Users\ziyan\OneDrive\Desktop\PGM_project\outputs\p_series_attention_sweep\2026-05-28_P-noPGM_p_traj_no_pgm_lambdanone_alphafalse_trajectory_matrix_attention_ebcc\checkpoints\best.pt
- last: C:\Users\ziyan\OneDrive\Desktop\PGM_project\outputs\p_series_attention_sweep\2026-05-28_P-noPGM_p_traj_no_pgm_lambdanone_alphafalse_trajectory_matrix_attention_ebcc\checkpoints\last.pt

## Notes

- What worked:
- What failed:
- What to try next: Attention-head lambda sweep copied from P3_trajectory_matrix_linear_noPGM.yaml. Same PGM placement/lambda as the source config; classifier is direct trajectory_matrix_attention over DiffNet relation tokens R. No InformationMatrixAccumulator.
