"""Reusable training utilities for precomputed embedding experiments."""

from __future__ import annotations

import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import EmbeddingDifferencePGMModel
from utils.experiment_manager import (
    ABLATION_INFO,
    append_registry,
    append_train_log,
    create_run_dir,
    get_ablation_id,
    get_model_variant,
    init_train_log,
    now_timestamp,
    save_metrics,
    save_resolved_config,
    write_experiment_card,
    write_model_summary,
)


VARIANT_TO_ABLATION = {info["variant"]: ablation_id for ablation_id, info in ABLATION_INFO.items()}


def set_controlled_variant(config: dict[str, Any], ablation_id: str | None = None, model_variant: str | None = None) -> None:
    if ablation_id is None and model_variant is None:
        return
    if model_variant is not None:
        if model_variant not in VARIANT_TO_ABLATION:
            valid = ", ".join(sorted(VARIANT_TO_ABLATION))
            raise ValueError(f"model_variant must be one of: {valid}")
        expected_ablation = VARIANT_TO_ABLATION[model_variant]
        if ablation_id is not None and ablation_id != expected_ablation:
            raise ValueError(f"{model_variant} corresponds to {expected_ablation}, not {ablation_id}")
        ablation_id = expected_ablation
    assert ablation_id is not None
    if ablation_id not in ABLATION_INFO:
        valid = ", ".join(sorted(ABLATION_INFO))
        raise ValueError(f"ablation_id must be one of: {valid}")

    variant = ABLATION_INFO[ablation_id]["variant"]
    config.setdefault("model", {})["ablation_id"] = ablation_id
    config["model"]["variant"] = variant
    info_config = config.setdefault("information_matrix", {})
    if ablation_id in {"E0", "E1"}:
        config.setdefault("pgm_smoother", {})["type"] = "none"
        config["pgm_smoother"]["lambda_smooth"] = None
        info_config["enabled"] = False
        info_config["use_alpha"] = False
    elif ablation_id in {"E2", "E3", "E4"}:
        config.setdefault("pgm_smoother", {})["type"] = "gaussian_chain"
        if config["pgm_smoother"].get("lambda_smooth") is None:
            config["pgm_smoother"]["lambda_smooth"] = 1.0
        info_config["enabled"] = ablation_id in {"E3", "E4"}
        if ablation_id == "E2":
            info_config["use_alpha"] = False
        else:
            info_config.setdefault("use_alpha", True)
    if ablation_id == "E4":
        config.setdefault("classifier", {})["type"] = "attention_pool"
    else:
        config.setdefault("classifier", {})["type"] = "mlp"


def str_to_bool(value: str | bool) -> bool:
    """Parse common CLI boolean spellings."""

    if isinstance(value, bool):
        return value
    normalized = value.lower().strip()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def torch_load(path: Path) -> dict[str, Any]:
    """Load a torch payload while supporting older PyTorch versions."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dictionary payload")
    return payload


def load_embedding_payload(path: Path) -> dict[str, Any]:
    """Load and validate an embedding file."""

    payload = torch_load(path)
    required_keys = {"X", "labels", "label_names", "metadata"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")

    X = payload["X"]
    labels = payload["labels"]
    if not isinstance(X, torch.Tensor) or X.ndim != 3:
        raise ValueError(f"{path} X must be a tensor with shape [N, T, D]")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"{path} labels must be a tensor with shape [N]")
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"{path} X and labels disagree on N: {X.shape[0]} vs {labels.shape[0]}")
    if not torch.isfinite(X).all():
        raise FloatingPointError(f"{path} X contains NaN or Inf values")

    payload["X"] = X.float()
    payload["labels"] = labels.long()
    return payload


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def prepare_training_config(
    base_config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    run_dir: Path | None,
    ablation_id: str | None = None,
    model_variant: str | None = None,
    pgm_type: str | None = None,
    classifier_type: str | None = None,
    lambda_smooth: float | None = None,
    use_alpha: bool | None = None,
    epochs: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a deep-copied config with run-specific overrides applied."""

    config = deepcopy(base_config)
    config.setdefault("training", {})
    config.setdefault("dataset", {})
    config.setdefault("output", {})
    config.setdefault("backbone", {})
    config.setdefault("diff_nn", {})
    config.setdefault("pgm_smoother", {})
    config.setdefault("information_matrix", {})
    config.setdefault("classifier", {})

    set_controlled_variant(config, ablation_id=ablation_id, model_variant=model_variant)
    if pgm_type is not None:
        if pgm_type not in {"none", "gaussian_chain", "learnable_gaussian_chain"}:
            raise ValueError("pgm_type must be none, gaussian_chain, or learnable_gaussian_chain")
        config["pgm_smoother"]["type"] = pgm_type
    if classifier_type is not None:
        if classifier_type not in {"mlp", "attention_pool"}:
            raise ValueError("classifier_type must be mlp or attention_pool")
        config["classifier"]["type"] = classifier_type
    if use_alpha is not None:
        config["information_matrix"]["use_alpha"] = use_alpha
    if lambda_smooth is not None:
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        config["pgm_smoother"]["lambda_smooth"] = float(lambda_smooth)
        config["pgm_smoother"].setdefault("type", "gaussian_chain")
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    if device is not None:
        config["training"]["device"] = device
    if batch_size is not None:
        config["training"]["batch_size"] = int(batch_size)
    if lr is not None:
        config["training"]["lr"] = float(lr)
    if weight_decay is not None:
        config["training"]["weight_decay"] = float(weight_decay)
    if seed is not None:
        config["training"]["seed"] = int(seed)

    config["dataset"]["train_file"] = str(train_file)
    config["dataset"]["val_file"] = str(val_file)
    if run_dir is not None:
        config["output"]["run_dir"] = str(run_dir)
    return config


def subset_payload(payload: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "X": payload["X"][:count].clone(),
        "labels": payload["labels"][:count].clone(),
        "label_names": payload.get("label_names", []),
        "metadata": payload.get("metadata", [])[:count],
    }


def make_loader(
    X: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(X, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for X, labels in loader:
        X = X.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(X)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def save_yaml(config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def train_from_config(
    config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    run_dir: Path | None,
    overfit_samples: int | None = None,
) -> dict[str, Any]:
    """Train one model configuration on precomputed embeddings."""

    training_config = config["training"]
    diff_type = config.get("diff_nn", {}).get("type", "pairwise_diff_net")
    if diff_type != "pairwise_diff_net":
        raise NotImplementedError('only diff_nn.type="pairwise_diff_net" is implemented')

    seed = int(training_config.get("seed", 0))
    set_reproducible_seed(seed)

    train_payload = load_embedding_payload(train_file)
    val_payload = load_embedding_payload(val_file)
    if overfit_samples is not None:
        train_payload = subset_payload(train_payload, overfit_samples)
        val_payload = subset_payload(train_payload, overfit_samples)

    D = int(train_payload["X"].shape[-1])
    config.setdefault("backbone", {})["input_dim"] = D
    if val_payload["X"].shape[-1] != D:
        raise ValueError(f"train D={D} but val D={val_payload['X'].shape[-1]}")

    if run_dir is not None:
        config.setdefault("output", {})["run_dir"] = str(run_dir)
    run_paths = create_run_dir(config)
    config.setdefault("output", {})["checkpoint_best"] = str(run_paths.best_checkpoint_path)
    config["output"]["checkpoint_last"] = str(run_paths.last_checkpoint_path)
    save_resolved_config(config, run_paths.run_dir)
    write_model_summary(config, run_paths.run_dir)
    init_train_log(run_paths.run_dir)

    batch_size = int(training_config.get("batch_size", 32))
    device = resolve_device(str(training_config.get("device", "cpu")))
    train_loader = make_loader(
        train_payload["X"],
        train_payload["labels"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        val_payload["X"],
        val_payload["labels"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = EmbeddingDifferencePGMModel.from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )

    epochs = int(training_config.get("epochs", 30))
    save_best = bool(training_config.get("save_best", True))
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    wandb_run = None
    logging_config = config.get("logging", {})
    if logging_config.get("use_wandb", False):
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=logging_config.get("wandb_project", "diving48-diff-pgm-info"),
                name=run_paths.run_name,
                config=config,
            )
        except ImportError:
            print("W&B requested but wandb is not installed; continuing with local logging only.")

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device)
        epoch_time_sec = time.perf_counter() - start_time
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["acc"],
            "lr": lr,
            "epoch_time_sec": epoch_time_sec,
        }
        history.append(row)
        append_train_log(run_paths.run_dir, row)
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_top1": val_metrics["acc"],
                "val_loss": val_metrics["loss"],
            },
            run_paths.last_checkpoint_path,
        )

        improved = (
            val_metrics["acc"] > best_val_acc
            or (val_metrics["acc"] == best_val_acc and val_metrics["loss"] < best_val_loss)
        )
        if improved:
            best_val_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            if save_best:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "best_val_top1": best_val_acc,
                        "best_val_loss": best_val_loss,
                    },
                    run_paths.best_checkpoint_path,
                )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "train/top1": train_metrics["acc"],
                    "val/top1": val_metrics["acc"],
                    "lr": lr,
                    "epoch": epoch,
                }
            )

    final = history[-1]
    ablation_id = get_ablation_id(config)
    model_variant = get_model_variant(config)
    pgm_type = config.get("pgm_smoother", {}).get("type")
    lambda_smooth = None if pgm_type == "none" else config.get("pgm_smoother", {}).get("lambda_smooth")
    info_enabled = bool(config.get("information_matrix", {}).get("enabled", ablation_id in {"E3", "E4"}))
    use_alpha = bool(config.get("information_matrix", {}).get("use_alpha", False)) if info_enabled else False
    metrics = {
        "run_name": run_paths.run_name,
        "date": now_timestamp(),
        "ablation_id": ablation_id,
        "model_variant": model_variant,
        "best_val_top1": best_val_acc,
        "best_val_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_top1": final["train_top1"],
        "final_train_loss": final["train_loss"],
        "final_val_top1": final["val_top1"],
        "final_val_loss": final["val_loss"],
        "test_loss": None,
        "test_top1": None,
        "lambda_smooth": lambda_smooth,
        "use_alpha": use_alpha,
        "classifier_type": config.get("classifier", {}).get("type"),
        "checkpoint_best": str(run_paths.best_checkpoint_path if save_best else ""),
        "checkpoint_last": str(run_paths.last_checkpoint_path),
        "notes": config.get("notes", ""),
        "seed": seed,
        "history": history,
        # Backward-compatible aliases for existing summary/grid helpers.
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_acc": final["train_top1"],
        "final_val_acc": final["val_top1"],
        "checkpoint_path": str(run_paths.best_checkpoint_path if save_best else ""),
    }
    save_metrics(run_paths.run_dir, metrics)
    write_experiment_card(config, metrics, run_paths.run_dir)
    append_registry(config, metrics, run_paths.run_dir)
    if wandb_run is not None:
        wandb_run.finish()
    return metrics


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
