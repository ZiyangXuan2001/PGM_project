"""RunPod launch helper for full controlled DiffTraj-PGM training."""

from __future__ import annotations

import argparse
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from check_runpod_environment import run_model_checks  # noqa: E402
from run_small_ablation import VARIANT_SPECS, print_summary, resolve_variants  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.experiment_manager import ABLATION_INFO  # noqa: E402
from utils.training import prepare_training_config, resolve_device, train_from_config  # noqa: E402


VARIANT_TO_ABLATION = {info["variant"]: ablation_id for ablation_id, info in ABLATION_INFO.items()}
VARIANT_TO_ABLATION.update(
    {
        "mean_pool_baseline": "E0",
        "diff_only": "E1",
        "diff_pgm": "E2",
        "diff_pgm_info": "E3",
        "diff_pgm_info_attention": "E3",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full RunPod training workflows.")
    parser.add_argument("--stage", choices=["check", "train", "ablation", "all"], default="check")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("/workspace/data/diving48_embeddings"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--variant", default="diff_pgm_info_accum")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument(
        "--frame-pgm-type",
        choices=["none", "gaussian_chain", "learnable_gaussian_chain"],
        default=None,
        help="Optional frame-side PGM smoother applied before DiffNet.",
    )
    parser.add_argument(
        "--frame-lambda-smooth",
        type=float,
        default=None,
        help="Frame-side Gaussian chain smoothness lambda.",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["mlp", "attention_pool", "temporal_evidence_attention"],
        default=None,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ablation_from_variant(value: str) -> str:
    normalized = value.strip()
    upper = normalized.upper()
    if upper == "E4":
        return "E3"
    if upper in ABLATION_INFO:
        return upper
    if normalized in VARIANT_TO_ABLATION:
        return VARIANT_TO_ABLATION[normalized]
    valid_variants = ", ".join(sorted(VARIANT_TO_ABLATION))
    valid_ids = ", ".join(sorted(ABLATION_INFO))
    raise SystemExit(f"--variant must be one of IDs ({valid_ids}) or variants ({valid_variants}).")


def candidate_file(
    embeddings_dir: Path,
    explicit_path: Path | None,
    config: dict[str, Any],
    config_key: str,
    candidates: list[str],
) -> Path:
    if explicit_path is not None:
        return resolve_path(explicit_path)

    dataset_config = config.get("dataset", {})
    configured = dataset_config.get(config_key)
    if configured:
        path = Path(configured)
        possible = path if path.is_absolute() else embeddings_dir / path
        if possible.is_file():
            return possible

    for name in candidates:
        possible = embeddings_dir / name
        if possible.is_file():
            return possible

    searched = [str(embeddings_dir / name) for name in candidates]
    if configured:
        searched.insert(0, str(embeddings_dir / str(configured)))
    raise FileNotFoundError(
        f"Could not find {config_key}. Checked:\n  " + "\n  ".join(searched)
    )


def resolve_embedding_files(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path]:
    embeddings_dir = resolve_path(args.embeddings_dir)
    train_file = candidate_file(
        embeddings_dir,
        args.train_file,
        config,
        "train_file",
        ["train_embeddings.pt", "train.pt"],
    )
    val_file = candidate_file(
        embeddings_dir,
        args.val_file,
        config,
        "val_file",
        ["val_embeddings.pt", "val.pt", "test_embeddings.pt", "test.pt"],
    )
    return train_file, val_file


def apply_output_root(config: dict[str, Any], output_root: Path | None) -> None:
    if output_root is None:
        return
    config.setdefault("output", {})["root"] = str(resolve_path(output_root))


def configure_full_run(
    base_config: dict[str, Any],
    ablation_id: str,
    train_file: Path,
    val_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = VARIANT_SPECS[ablation_id]
    config = prepare_training_config(
        base_config=deepcopy(base_config),
        train_file=train_file,
        val_file=val_file,
        run_dir=None,
        ablation_id=ablation_id,
        pgm_type=spec["pgm_type"],
        frame_pgm_type=args.frame_pgm_type,
        classifier_type=args.classifier_type or spec["classifier_type"],
        lambda_smooth=spec["lambda_smooth"],
        frame_lambda_smooth=args.frame_lambda_smooth,
        use_alpha=spec["use_alpha"],
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        progress_every=args.progress_every,
        seed=args.seed,
    )
    config.setdefault("information_matrix", {})["enabled"] = bool(spec["information_enabled"])
    config["information_matrix"]["use_alpha"] = bool(spec["use_alpha"])
    config.setdefault("classifier", {})["type"] = args.classifier_type or spec["classifier_type"]
    config.setdefault("pgm_smoother", {})["type"] = spec["pgm_type"]
    config["pgm_smoother"]["lambda_smooth"] = spec["lambda_smooth"]
    config.setdefault("notes", "RunPod full training launcher run.")
    apply_output_root(config, args.output_root)
    return config


def run_check(config: dict[str, Any], device_name: str) -> None:
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"/workspace exists: {Path('/workspace').exists()}")
    device = resolve_device(device_name)
    print(f"Selected device: {device}")
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory writable: {output_dir}")
    run_model_checks(config, device)


def run_training(args: argparse.Namespace, config: dict[str, Any], ablation_ids: list[str]) -> list[dict[str, Any]]:
    train_file, val_file = resolve_embedding_files(args, config)
    print(f"Train embeddings: {train_file}")
    print(f"Val embeddings: {val_file}")

    rows: list[dict[str, Any]] = []
    for ablation_id in ablation_ids:
        print(f"\n=== Running {ablation_id}: {ABLATION_INFO[ablation_id]['variant']} ===")
        run_config = configure_full_run(config, ablation_id, train_file, val_file, args)
        metrics = train_from_config(
            config=run_config,
            train_file=train_file,
            val_file=val_file,
            run_dir=None,
        )
        rows.append(
            {
                "ablation_id": metrics["ablation_id"],
                "model_variant": metrics["model_variant"],
                "best_val_top1": metrics["best_val_top1"],
                "best_val_epoch": metrics["best_val_epoch"],
                "final_train_loss": metrics["final_train_loss"],
                "final_val_loss": metrics["final_val_loss"],
                "run_dir": run_config.get("output", {}).get("run_dir", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))

    if args.stage in {"check", "all"}:
        print("RunPod full launcher check stage")
        run_check(config, args.device)

    if args.stage == "train":
        ablation_id = ablation_from_variant(args.variant)
        rows = run_training(args, config, [ablation_id])
        print_summary(rows)
    elif args.stage in {"ablation", "all"}:
        ablation_ids = resolve_variants(args.variants)
        rows = run_training(args, config, ablation_ids)
        print_summary(rows)


if __name__ == "__main__":
    main()
