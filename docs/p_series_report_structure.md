# P-Series Report Structure

## 1. Abstract

- One-paragraph summary of the problem, method, and main finding.
- Suggested claim: on frozen CLIP ViT-B/16 frame embeddings for Diving48, Gaussian
  temporal smoothing improves the simplified trajectory-matrix baseline, with the
  strongest current multi-seed result from post-trajectory PGM.

## 2. Introduction

- Motivation: frozen image features miss some temporal consistency needed for
  fine-grained diving action classification.
- Research question: does a lightweight Gaussian-chain PGM help by smoothing
  either raw frame features or trajectory features?
- Contributions:
  - Simplified P-series model family without InformationMatrixAccumulator.
  - Pre-trajectory and post-trajectory PGM ablations.
  - Direct attention classifier follow-up on the two best PGM placements.
  - Lambda sweep plus multi-seed validation for the key comparison.

## 3. Data And Setup

- Dataset: Diving48, 48 action classes.
- Input representation: frozen CLIP ViT-B/16 pooled frame features.
- Tensor shape: `X: [B, 16, 512]`.
- Evaluation: validation top-1 accuracy and validation loss.
- Keep optimizer, batch size, feature path, split, epochs, seed protocol, and
  classifier fixed across ablations except for the intended PGM/head variables.

## 4. Method

### 4.1 Baseline Trajectory Model

- `X -> TemporalProjection -> DiffNet -> R -> trajectory_matrix_linear -> logits`.
- No attention classifier.
- No InformationMatrixAccumulator.

### 4.2 Gaussian-Chain PGM

- Energy:
  `alpha/2 * sum_t ||Z_t - X_t||^2 + lambda/2 * sum_t ||Z_{t+1} - Z_t||^2`.
- MAP solve:
  `(alpha I + lambda L) Z = alpha X`.
- Fixed PGM has zero trainable parameters.

### 4.3 PGM Placement

- No PGM: `X -> TemporalProjection -> DiffNet -> R -> classifier`.
- PrePGM: `X -> PGM -> TemporalProjection -> DiffNet -> R -> classifier`.
- PostPGM: `X -> TemporalProjection -> PGM -> DiffNet -> R -> classifier`.

### 4.4 Direct Attention Head Follow-Up

- Keep the same two best PGM placements:
  - P12: raw-frame PrePGM with `lambda=0.30`.
  - P17: projected-frame PostPGM with `lambda=0.80`.
- Replace the flattened linear classifier with direct attention over relation
  tokens:
  `R [B, 15, d_r] -> trajectory_matrix_attention -> logits`.
- This is not the InformationMatrixAccumulator. It is a plain classifier head
  over the temporal DiffNet relation sequence.

## 5. Experiments

### 5.1 Main Comparison

- P3: noPGM trajectory_matrix_linear.
- P4/P5 and expanded postPGM sweep: postPGM with different lambdas.
- P6/P7 and expanded prePGM sweep: prePGM with different lambdas.

### 5.2 Lambda Sweep

- Compare prePGM and postPGM across lambda values.
- Use plots:
  - `outputs/analysis/p_series_postpgm_lambda_sweep_full.png`
  - `outputs/analysis/p_series_prepgm_lambda_sweep_full.png`
  - `outputs/analysis/p_series_pgm_layer_lambda_comparison_full.png`

### 5.3 Multi-Seed Confirmation

- Core runs:
  - P3 noPGM.
  - P12 prePGM lambda=0.30.
  - P17 postPGM lambda=0.80.
- Use seeds 0, 1, and 2.
- Artifacts:
  - `outputs/analysis/p_series_core_multiseed_comparison.png`
  - `outputs/analysis/p_series_core_multiseed_summary.csv`
  - `outputs/analysis/p_series_core_multiseed_aggregate.csv`

### 5.4 Attention-Head Follow-Up

- Run only the two best PGM placements with the new classifier:
  - `configs/p_series/P12_prePGM_lam030_trajectory_matrix_attention.yaml`
  - `configs/p_series/P17_trajectory_matrix_attention_pgm_lam080.yaml`
- Main comparison: does direct attention over `R` improve over flattening `R`?

## 6. Results

Report the main multi-seed table:

| Model | Mean Best Val Top-1 | Std | Best Seed |
|---|---:|---:|---:|
| P3 noPGM | 0.1527 | 0.0044 | 0.1565 |
| P12 prePGM lambda=0.30 | 0.1616 | 0.0043 | 0.1665 |
| P17 postPGM lambda=0.80 | 0.1651 | 0.0041 | 0.1689 |

Interpretation:

- Both PGM placements beat noPGM in the current 3-seed comparison.
- PostPGM lambda=0.80 is the strongest current result.
- PrePGM helps too, but slightly less than postPGM in this setup.

## 7. Discussion

- Why postPGM may work: trajectory features are already motion-like and may be
  easier to smooth without erasing raw frame differences.
- Why prePGM can help: it denoises frozen CLIP frame observations before
  trajectory extraction.
- Why too much smoothing can hurt: it can suppress useful temporal change.
- Discuss overfitting: training loss falls strongly while validation loss rises
  after the best epoch, so early stopping is important.

## 8. Limitations

- Absolute top-1 is still modest.
- Experiments use frozen CLIP features rather than end-to-end video training.
- Main confirmation currently uses 3 seeds, enough for a class report but not a
  final statistical claim.
- No held-out test-set claim unless a separate test protocol is run.

## 9. Conclusion

- The simplified P-series ablation supports the claim that fixed Gaussian-chain
  PGM smoothing can improve trajectory-matrix classification.
- Best current evidence: post-trajectory smoothing at lambda=0.80 improves over
  the noPGM baseline across 3 seeds.

## 10. Appendix

- Config paths for all P-series runs.
- Parameter counts.
- Full training curves.
- Logs and checkpoint paths.
- Notes on failed partial runs and successful retries.
