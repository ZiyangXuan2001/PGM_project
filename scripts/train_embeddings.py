"""Train the controlled DiffTraj-PGM model on precomputed frame embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.training import prepare_training_config, str_to_bool, train_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on saved embedding tensors.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument("--ablation_id", choices=["E0", "E1", "E1.5", "E2", "E3", "E4"], default=None)
    parser.add_argument(
        "--model_variant",
        choices=[
            "feature_mean",
            "diff_mean",
            "diff_info_accum",
            "diff_pgm_mean",
            "diff_pgm_info_accum",
            "mean_pool_baseline",
            "diff_only",
            "diff_pgm",
            "diff_pgm_info",
            "diff_pgm_info_attention",
        ],
        default=None,
    )
    parser.add_argument("--pgm_type", choices=["none", "gaussian_chain", "learnable_gaussian_chain"], default=None)
    parser.add_argument("--classifier_type", choices=["mlp", "attention_pool"], default=None)
    parser.add_argument("--lambda_smooth", type=float, default=None)
    parser.add_argument("--use_alpha", type=str_to_bool, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--overfit_samples", type=int, default=None)
    return parser.parse_args()


def path_from_config(config: dict, key: str) -> Path:
    dataset_config = config.get("dataset", {})
    value = dataset_config.get(key)
    if value is None:
        raise SystemExit(f"--{key} is required when dataset.{key} is not set in config.")
    path = Path(value)
    if path.is_absolute():
        return path
    embedding_root = dataset_config.get("embedding_root")
    if embedding_root is not None:
        return PROJECT_ROOT / Path(embedding_root) / path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))

    train_file = args.train_file or path_from_config(config, "train_file")
    val_file = args.val_file or path_from_config(config, "val_file")
    run_dir = args.run_dir

    config = prepare_training_config(
        base_config=config,
        train_file=train_file,
        val_file=val_file,
        run_dir=run_dir,
        ablation_id=args.ablation_id,
        model_variant=args.model_variant,
        pgm_type=args.pgm_type,
        classifier_type=args.classifier_type,
        lambda_smooth=args.lambda_smooth,
        use_alpha=args.use_alpha,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    if args.overfit_samples is not None:
        config.setdefault("training", {})["overfit_samples"] = args.overfit_samples

    metrics = train_from_config(
        config=config,
        train_file=train_file,
        val_file=val_file,
        run_dir=run_dir,
        overfit_samples=args.overfit_samples,
    )

    summary = {
        "best_epoch": metrics["best_epoch"],
        "best_val_acc": metrics["best_val_acc"],
        "best_val_loss": metrics["best_val_loss"],
        "final_train_acc": metrics["final_train_acc"],
        "final_train_loss": metrics["final_train_loss"],
        "checkpoint_path": metrics["checkpoint_path"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
