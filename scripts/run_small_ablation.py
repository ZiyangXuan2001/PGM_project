"""Run tiny local E0-E4 training checks on fake or saved CLIP embeddings."""

from __future__ import annotations

import argparse
import copy
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.training import prepare_training_config, set_reproducible_seed, train_from_config  # noqa: E402


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "E0": {
        "variant": "mean_pool_baseline",
        "pgm_type": "none",
        "classifier_type": "mlp",
        "information_enabled": False,
        "use_alpha": False,
        "lambda_smooth": None,
    },
    "E1": {
        "variant": "diff_only",
        "pgm_type": "none",
        "classifier_type": "mlp",
        "information_enabled": False,
        "use_alpha": False,
        "lambda_smooth": None,
    },
    "E2": {
        "variant": "diff_pgm",
        "pgm_type": "gaussian_chain",
        "classifier_type": "mlp",
        "information_enabled": False,
        "use_alpha": False,
        "lambda_smooth": 1.0,
    },
    "E3": {
        "variant": "diff_pgm_info",
        "pgm_type": "gaussian_chain",
        "classifier_type": "mlp",
        "information_enabled": True,
        "use_alpha": True,
        "lambda_smooth": 1.0,
    },
    "E4": {
        "variant": "diff_pgm_info_attention",
        "pgm_type": "gaussian_chain",
        "classifier_type": "attention_pool",
        "information_enabled": True,
        "use_alpha": True,
        "lambda_smooth": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run small E0-E4 training checks without downloading Diving48."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--variants", default="all", help='Use "all" or a comma list such as "E0,E2".')
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for outputs/runs; useful for temporary launcher tests.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional override for experiments/experiment_registry.csv.",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=None,
        help="Where to store the tiny train/val tensors used by this launcher.",
    )
    return parser.parse_args()


def resolve_variants(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["E0", "E1", "E2", "E3", "E4"]
    variants = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not variants:
        raise SystemExit("--variants must be 'all' or a comma-separated list like E0,E2")
    invalid = [variant for variant in variants if variant not in VARIANT_SPECS]
    if invalid:
        valid = ", ".join(VARIANT_SPECS)
        raise SystemExit(f"Unknown variant(s): {', '.join(invalid)}. Valid variants: {valid}")
    return variants


def resolve_device_name(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_load_any(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_real_embeddings(path: Path) -> dict[str, Any]:
    """Load real precomputed embeddings from supported local .pt formats."""

    if not path.exists():
        raise FileNotFoundError(f"embedding file not found: {path}")
    payload = torch_load_any(path)

    label_names: list[str] | None = None
    metadata: list[Any] = []
    if isinstance(payload, dict):
        if "embeddings" in payload:
            X = payload["embeddings"]
        elif "X" in payload:
            X = payload["X"]
        else:
            raise ValueError(
                f"{path} dict format must contain 'embeddings' or 'X' with shape [N, T, 512]."
            )
        labels = payload.get("labels")
        if labels is None:
            raise ValueError(f"{path} dict format must contain 'labels' with shape [N].")
        label_names = payload.get("label_names")
        video_ids = payload.get("video_ids")
        if video_ids is not None:
            metadata = [{"video_id": str(video_id)} for video_id in video_ids]
        else:
            metadata = list(payload.get("metadata", []))
    elif isinstance(payload, list):
        embeddings = []
        labels_list = []
        metadata = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict) or "embedding" not in item or "label" not in item:
                raise ValueError(
                    f"{path} list format expects dicts with 'embedding' [T, 512] and 'label'; "
                    f"bad item at index {index}."
                )
            embeddings.append(torch.as_tensor(item["embedding"]).float())
            labels_list.append(int(item["label"]))
            metadata.append({"video_id": str(item.get("video_id", index))})
        X = torch.stack(embeddings, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
    else:
        raise ValueError(
            f"{path} has unsupported format. Expected a dict with embeddings/labels "
            "or a list of {'embedding', 'label', 'video_id'} dicts."
        )

    if not isinstance(X, torch.Tensor) or X.ndim not in {3, 4}:
        raise ValueError(f"{path} embeddings must be a tensor with shape [N, T, D] or [N, T, S, D].")
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"{path} embeddings and labels disagree on N: {X.shape[0]} vs {labels.shape[0]}")
    if not torch.isfinite(X).all():
        raise FloatingPointError(f"{path} embeddings contain NaN or Inf values.")

    if label_names is None:
        label_names = [str(index) for index in range(48)]
    if len(metadata) != X.shape[0]:
        metadata = [{"index": index} for index in range(X.shape[0])]

    return {
        "X": X.float(),
        "labels": labels.long(),
        "label_names": label_names,
        "metadata": metadata,
        "backbone_name": payload.get("backbone_name") if isinstance(payload, dict) else None,
        "feature_format": payload.get("feature_format") if isinstance(payload, dict) else None,
        "embedding_dim": int(X.shape[-1]),
    }


def make_fake_payload(config: dict[str, Any], max_samples: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_frames = int(config.get("backbone", {}).get("num_frames", 16))
    input_dim = int(config.get("backbone", {}).get("input_dim", 512))
    num_classes = int(config.get("dataset", {}).get("num_classes", 48))
    if config.get("backbone", {}).get("feature_format") == "spatial_map":
        spatial_tokens = int(config.get("backbone", {}).get("spatial_tokens", 49))
        X = torch.randn(max_samples, num_frames, spatial_tokens, input_dim, generator=generator)
    else:
        X = torch.randn(max_samples, num_frames, input_dim, generator=generator)
    labels = torch.randint(0, num_classes, (max_samples,), generator=generator)
    return {
        "X": X,
        "labels": labels,
        "label_names": [str(index) for index in range(num_classes)],
        "metadata": [{"source": "fake", "index": index} for index in range(max_samples)],
        "feature_format": config.get("backbone", {}).get("feature_format", "vector"),
        "embedding_dim": input_dim,
    }


def random_subset(payload: dict[str, Any], max_samples: int, seed: int) -> dict[str, Any]:
    total = payload["X"].shape[0]
    count = min(max_samples, total)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:count]
    return subset_by_indices(payload, indices)


def balanced_subset(
    payload: dict[str, Any],
    samples_per_class: int,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = payload["labels"]
    if samples_per_class <= 0:
        return random_subset(payload, max_samples=max_samples, seed=seed)

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_class.setdefault(int(label), []).append(index)
    if len(by_class) < 2:
        print("Warning: balanced sampling needs at least two classes; using random max-sample subset.")
        return random_subset(payload, max_samples=max_samples, seed=seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    selected: list[int] = []
    for label in sorted(by_class):
        class_indices = torch.tensor(by_class[label], dtype=torch.long)
        order = torch.randperm(class_indices.numel(), generator=generator)
        chosen = class_indices[order[:samples_per_class]].tolist()
        selected.extend(int(index) for index in chosen)

    if not selected:
        print("Warning: balanced sampling selected no items; using random max-sample subset.")
        return random_subset(payload, max_samples=max_samples, seed=seed)

    selected_tensor = torch.tensor(selected, dtype=torch.long)
    if selected_tensor.numel() > max_samples:
        order = torch.randperm(selected_tensor.numel(), generator=generator)[:max_samples]
        selected_tensor = selected_tensor[order]
    return subset_by_indices(payload, selected_tensor)


def subset_by_indices(payload: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    index_list = [int(index) for index in indices.tolist()]
    return {
        "X": payload["X"][indices].clone(),
        "labels": payload["labels"][indices].clone(),
        "label_names": copy.deepcopy(payload.get("label_names", [str(index) for index in range(48)])),
        "metadata": [copy.deepcopy(payload.get("metadata", [])[index]) for index in index_list]
        if payload.get("metadata")
        else [{"index": index} for index in index_list],
        "backbone_name": payload.get("backbone_name"),
        "feature_format": payload.get("feature_format"),
        "embedding_dim": int(payload["X"].shape[-1]),
    }


def split_payload(payload: dict[str, Any], seed: int, val_fraction: float = 0.2) -> tuple[dict[str, Any], dict[str, Any]]:
    total = payload["X"].shape[0]
    if total < 2:
        raise ValueError("Need at least two samples to create train/val splits.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)
    val_count = max(1, int(round(total * val_fraction)))
    val_count = min(val_count, total - 1)
    train_indices = indices[:-val_count]
    val_indices = indices[-val_count:]
    return subset_by_indices(payload, train_indices), subset_by_indices(payload, val_indices)


def make_group_subset_dir(args: argparse.Namespace) -> Path:
    if args.subset_dir is not None:
        subset_dir = args.subset_dir
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        subset_dir = PROJECT_ROOT / "outputs" / "small_ablation_inputs" / f"{stamp}_{uuid.uuid4().hex[:4]}"
    if not subset_dir.is_absolute():
        subset_dir = PROJECT_ROOT / subset_dir
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


def save_split_payloads(train_payload: dict[str, Any], val_payload: dict[str, Any], subset_dir: Path) -> tuple[Path, Path]:
    train_file = subset_dir / "train_subset.pt"
    val_file = subset_dir / "val_subset.pt"
    torch.save(train_payload, train_file)
    torch.save(val_payload, val_file)
    return train_file, val_file


def configure_variant(
    base_config: dict[str, Any],
    variant_id: str,
    train_file: Path,
    val_file: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    spec = VARIANT_SPECS[variant_id]
    config = prepare_training_config(
        base_config=base_config,
        train_file=train_file,
        val_file=val_file,
        run_dir=None,
        ablation_id=variant_id,
        pgm_type=spec["pgm_type"],
        classifier_type=spec["classifier_type"],
        lambda_smooth=spec["lambda_smooth"],
        use_alpha=spec["use_alpha"],
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0 if args.overfit else None,
        seed=args.seed,
    )
    config.setdefault("information_matrix", {})["enabled"] = bool(spec["information_enabled"])
    config["information_matrix"]["use_alpha"] = bool(spec["use_alpha"])
    config.setdefault("classifier", {})["type"] = spec["classifier_type"]
    config.setdefault("pgm_smoother", {})["type"] = spec["pgm_type"]
    config["pgm_smoother"]["lambda_smooth"] = spec["lambda_smooth"]
    if args.overfit:
        config.setdefault("diff_nn", {})["dropout"] = 0.0
        config.setdefault("classifier", {})["dropout"] = 0.0
        config.setdefault("notes", "Small overfit debugging run.")
    else:
        config.setdefault("notes", f"Small {args.mode} ablation launcher run.")
    if args.output_root is not None:
        config.setdefault("output", {})["root"] = str(args.output_root)
    if args.registry_path is not None:
        config.setdefault("output", {})["registry_path"] = str(args.registry_path)
    return config


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSmall ablation summary")
    print(
        f"{'ID':<3} {'variant':<28} {'best_val_top1':>13} "
        f"{'best_epoch':>10} {'train_loss':>12} {'val_loss':>12} run_dir"
    )
    for row in rows:
        print(
            f"{row['ablation_id']:<3} "
            f"{row['model_variant']:<28} "
            f"{float(row['best_val_top1']):>13.4f} "
            f"{int(row['best_val_epoch']):>10d} "
            f"{float(row['final_train_loss']):>12.4f} "
            f"{float(row['final_val_loss']):>12.4f} "
            f"{row['run_dir']}"
        )


def main() -> None:
    args = parse_args()
    args.max_samples = int(args.max_samples if args.max_samples is not None else (32 if args.overfit else 256))
    args.epochs = int(args.epochs if args.epochs is not None else (20 if args.overfit else 3))
    if args.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    set_reproducible_seed(args.seed)
    variants = resolve_variants(args.variants)
    device = resolve_device_name(args.device)
    base_config = load_config(str(args.config))

    if args.mode == "fake":
        payload = make_fake_payload(base_config, max_samples=args.max_samples, seed=args.seed)
    else:
        if args.embeddings_path is None:
            raise SystemExit("--embeddings-path is required when --mode real.")
        real_payload = load_real_embeddings(args.embeddings_path)
        payload = balanced_subset(
            real_payload,
            samples_per_class=args.samples_per_class,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        if payload["X"].shape[0] > args.max_samples:
            payload = random_subset(payload, max_samples=args.max_samples, seed=args.seed)
        if payload["X"].shape[0] < 2:
            raise SystemExit("Real embedding subset has fewer than two samples.")

    train_payload, val_payload = split_payload(payload, seed=args.seed)
    subset_dir = make_group_subset_dir(args)
    train_file, val_file = save_split_payloads(train_payload, val_payload, subset_dir)

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Variants: {', '.join(variants)}")
    print(f"Train samples: {train_payload['X'].shape[0]} | Val samples: {val_payload['X'].shape[0]}")
    print(f"Small split tensors: {subset_dir}")

    summary_rows: list[dict[str, Any]] = []
    for variant_id in variants:
        print(f"\n=== Running {variant_id}: {VARIANT_SPECS[variant_id]['variant']} ===")
        config = configure_variant(
            base_config=base_config,
            variant_id=variant_id,
            train_file=train_file,
            val_file=val_file,
            args=args,
            device=device,
        )
        metrics = train_from_config(
            config=config,
            train_file=train_file,
            val_file=val_file,
            run_dir=None,
        )
        summary_rows.append(
            {
                "ablation_id": metrics["ablation_id"],
                "model_variant": metrics["model_variant"],
                "best_val_top1": metrics["best_val_top1"],
                "best_val_epoch": metrics["best_val_epoch"],
                "final_train_loss": metrics["final_train_loss"],
                "final_val_loss": metrics["final_val_loss"],
                "run_dir": config.get("output", {}).get("run_dir", ""),
            }
        )

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
