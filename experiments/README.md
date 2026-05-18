# Experiments

This folder tracks local experiment metadata for the controlled Diving48 V2
DiffTraj-PGM ablation plan. Runs are logged locally under `outputs/runs/` and
summarized in `experiments/experiment_registry.csv`.

| ID | Variant | Backbone | PGM Smooth | Representation | Classifier | Purpose |
|---|---|---|---|---|---|---|
| E0 | mean_pool_baseline | CLIP-B/16 | none | mean_pool(X) | MLP | basic frozen CLIP baseline |
| E1 | diff_only | CLIP-B/16 | none | mean_pool(R) | MLP | test PairwiseDiffNet |
| E2 | diff_pgm | CLIP-B/16 | gaussian_chain | mean_pool(Y) | MLP | test PGM smoothing |
| E3 | diff_pgm_info | CLIP-B/16 | gaussian_chain | information matrix H | MLP | test information matrix |
| E4 | diff_pgm_info_attention | CLIP-B/16 | gaussian_chain | information matrix H | attention head | test attention classifier |

Architecture definitions:

E0:
X -> mean_pool(X) -> MLP

E1:
X -> PairwiseDiffNet -> R -> mean_pool(R) -> MLP

E2:
X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> mean_pool(Y) -> MLP

E3:
X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> InformationMatrixAccumulator -> H -> MLP

E4:
X -> PairwiseDiffNet -> R -> GaussianPGMSmoother -> Y -> InformationMatrixAccumulator -> H -> AttentionHead

## Small training workflow

Step 1:
Run fake-data training to test model and logging.

Command:
python scripts/run_small_ablation.py --mode fake --max-samples 128 --epochs 2 --batch-size 16

Step 2:
Run real tiny embedding training.

Command:
python scripts/run_small_ablation.py --mode real --embeddings-path data/diving48_embeddings/train_embeddings.pt --samples-per-class 2 --epochs 3 --batch-size 16

Step 3:
Run tiny overfit test.

Command:
python scripts/run_small_ablation.py --mode real --embeddings-path data/diving48_embeddings/train_embeddings.pt --max-samples 64 --overfit --epochs 20 --batch-size 16

Fake mode checks code: forward, backward, optimizer, checkpointing, local logging, and registry updates.
Real tiny mode checks that the saved embedding file can be loaded and trained on.
Overfit mode checks whether the model can learn a very small subset before full Diving48 V2 training.
