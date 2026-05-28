"""Reusable training utilities for precomputed embedding experiments."""

from __future__ import annotations

import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset

from models import EmbeddingDifferencePGMModel, PSeriesTrajectoryMatrixModel
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
    legacy_variant_aliases = {
        "mean_pool_baseline": "feature_mean",
        "diff_only": "diff_mean",
        "diff_pgm": "diff_pgm_mean",
        "diff_pgm_info": "diff_pgm_info_accum",
        "diff_pgm_info_attention": "diff_pgm_info_accum",
    }
    legacy_ablation_aliases = {"E4": "E3"}
    if ablation_id is not None:
        ablation_id = legacy_ablation_aliases.get(str(ablation_id), str(ablation_id))
    if model_variant is not None:
        model_variant = legacy_variant_aliases.get(model_variant, model_variant)
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
    elif ablation_id == "E1.5":
        config.setdefault("pgm_smoother", {})["type"] = "none"
        config["pgm_smoother"]["lambda_smooth"] = None
        info_config["enabled"] = True
        info_config.setdefault("use_alpha", True)
    elif ablation_id in {"E2", "E3"}:
        config.setdefault("pgm_smoother", {})["type"] = "gaussian_chain"
        if config["pgm_smoother"].get("lambda_smooth") is None:
            config["pgm_smoother"]["lambda_smooth"] = 1.0
        info_config["enabled"] = ablation_id == "E3"
        if ablation_id == "E2":
            info_config["use_alpha"] = False
        else:
            info_config.setdefault("use_alpha", True)
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
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dictionary payload")
    return payload


def load_sidecar_tensor(path: Path, payload: dict[str, Any]) -> torch.Tensor | None:
    x_path = payload.get("X_path")
    if not isinstance(x_path, str):
        return None
    resolved = Path(x_path)
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"{path} references missing sidecar tensor file: {resolved}")
    array = np.load(resolved, mmap_mode="r")
    return torch.from_numpy(array)


def load_chunked_embedding_payload(
    path: Path,
    split_name: str,
    feature_layer: str | None = None,
) -> dict[str, Any]:
    """Load a chunk manifest as a lazy concatenated dataset."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"{path} must contain a 'chunks' list")
    if feature_layer is None:
        raise ValueError(f"{path} is a chunk manifest; set dataset.feature_layer or backbone.feature_layer")

    datasets: list[Dataset] = []
    labels_list: list[torch.Tensor] = []
    metadata: list[Any] = []
    label_names: list[str] | None = None
    expected_tail: tuple[int, ...] | None = None
    backbone_name: str | None = None
    feature_format = "spatial_map"

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(f"{path} contains a non-dict chunk entry")
        layers = chunk.get("layers")
        if not isinstance(layers, dict) or feature_layer not in layers:
            continue
        layer_entry = layers[feature_layer]
        if not isinstance(layer_entry, dict):
            continue
        split_entry = layer_entry.get(split_name)
        if split_entry is None:
            continue
        if not isinstance(split_entry, dict):
            raise ValueError(f"{path} chunk {chunk.get('name')} {feature_layer}.{split_name} is not a dict")
        pt_path = Path(split_entry.get("pt", ""))
        if not pt_path.is_absolute():
            pt_path = path.parent / pt_path
        payload = torch_load(pt_path)
        x_path = payload.get("X_path")
        if not isinstance(x_path, str):
            raise ValueError(f"{pt_path} is missing X_path")
        resolved_x = Path(x_path)
        if not resolved_x.is_absolute():
            resolved_x = pt_path.parent / resolved_x
        X = torch.from_numpy(np.load(resolved_x, mmap_mode="r"))
        labels = payload["labels"].long()
        if X.shape[0] != labels.shape[0]:
            raise ValueError(f"{pt_path} X and labels disagree on N: {X.shape[0]} vs {labels.shape[0]}")
        current_tail = tuple(int(v) for v in X.shape[1:])
        if expected_tail is None:
            expected_tail = current_tail
        elif current_tail != expected_tail:
            raise ValueError(f"{pt_path} shape tail {current_tail} differs from {expected_tail}")
        datasets.append(TensorDataset(X, labels))
        labels_list.append(labels)
        metadata.extend(payload.get("metadata", []))
        if label_names is None:
            label_names = list(payload.get("label_names", []))
        backbone_name = str(payload.get("backbone_name", f"resnet50_{feature_layer}"))
        feature_format = str(payload.get("feature_format", "spatial_map"))

    if not datasets or expected_tail is None:
        raise ValueError(f"{path} contains no chunks for feature_layer={feature_layer!r}, split={split_name!r}")

    labels_all = torch.cat(labels_list, dim=0)
    x_shape = (int(labels_all.shape[0]), *expected_tail)
    return {
        "dataset": ConcatDataset(datasets),
        "X_shape": x_shape,
        "labels": labels_all,
        "label_names": label_names or [str(index) for index in range(48)],
        "metadata": metadata,
        "backbone_name": backbone_name,
        "feature_format": feature_format,
        "selected_feature_layer": feature_layer,
        "manifest_path": str(path),
        "split_name": split_name,
    }


def load_embedding_payload(
    path: Path,
    feature_layer: str | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    """Load and validate an embedding file."""

    if path.suffix.lower() == ".json":
        return load_chunked_embedding_payload(path, split_name=split_name or "train", feature_layer=feature_layer)

    payload = torch_load(path)
    payload = dict(payload)
    sidecar_X = load_sidecar_tensor(path, payload)
    if sidecar_X is not None:
        payload["X"] = sidecar_X
        selected_layer = payload.get("selected_feature_layer")
        if feature_layer is not None and selected_layer is not None and feature_layer != selected_layer:
            raise ValueError(
                f"{path} contains feature_layer={selected_layer!r}, but config requested {feature_layer!r}"
            )

    if "X" not in payload:
        available_layers = sorted(key.removeprefix("X_") for key in payload if key.startswith("X_layer"))
        if feature_layer is None:
            raise ValueError(
                f"{path} is a multilayer embedding payload. Set dataset.feature_layer "
                f"or backbone.feature_layer to one of: {available_layers}"
            )
        layer_key = f"X_{feature_layer}"
        if layer_key not in payload:
            raise ValueError(
                f"{path} does not contain {layer_key}. Available layers: {available_layers}"
            )
        payload["X"] = payload[layer_key]
        payload["selected_feature_layer"] = feature_layer
        payload["feature_format"] = "spatial_map"
        if payload.get("backbone_name") == "resnet50":
            payload["backbone_name"] = f"resnet50_{feature_layer}"
        feature_shapes = payload.get("feature_shapes", {})
        if isinstance(feature_shapes, dict) and feature_layer in feature_shapes:
            payload["feature_shape"] = feature_shapes[feature_layer]

    required_keys = {"X", "labels", "label_names", "metadata"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")

    X = payload["X"]
    labels = payload["labels"]
    if not isinstance(X, torch.Tensor) or X.ndim not in {3, 4}:
        raise ValueError(f"{path} X must be a tensor with shape [N, T, D] or [N, T, S, D]")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"{path} labels must be a tensor with shape [N]")
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"{path} X and labels disagree on N: {X.shape[0]} vs {labels.shape[0]}")
    if sidecar_X is None and not torch.isfinite(X).all():
        raise FloatingPointError(f"{path} X contains NaN or Inf values")

    payload["X"] = X
    payload["labels"] = labels.long()
    payload["X_shape"] = tuple(int(v) for v in X.shape)
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
    frame_pgm_type: str | None = None,
    classifier_type: str | None = None,
    lambda_smooth: float | None = None,
    frame_lambda_smooth: float | None = None,
    use_alpha: bool | None = None,
    epochs: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    early_stop_patience: int | None = None,
    progress_every: int | None = None,
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
    config.setdefault("frame_pgm_smoother", {})
    config.setdefault("information_matrix", {})
    config.setdefault("classifier", {})

    set_controlled_variant(config, ablation_id=ablation_id, model_variant=model_variant)
    if pgm_type is not None:
        if pgm_type not in {"none", "gaussian_chain", "learnable_gaussian_chain"}:
            raise ValueError("pgm_type must be none, gaussian_chain, or learnable_gaussian_chain")
        config["pgm_smoother"]["type"] = pgm_type
    if frame_pgm_type is not None:
        if frame_pgm_type not in {"none", "gaussian_chain", "learnable_gaussian_chain"}:
            raise ValueError("frame_pgm_type must be none, gaussian_chain, or learnable_gaussian_chain")
        config["frame_pgm_smoother"]["type"] = frame_pgm_type
    if classifier_type is not None:
        if classifier_type not in {"mlp", "attention_pool", "temporal_evidence_attention"}:
            raise ValueError("classifier_type must be mlp, attention_pool, or temporal_evidence_attention")
        config["classifier"]["type"] = classifier_type
    if use_alpha is not None:
        config["information_matrix"]["use_alpha"] = use_alpha
    if lambda_smooth is not None:
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        config["pgm_smoother"]["lambda_smooth"] = float(lambda_smooth)
        config["pgm_smoother"].setdefault("type", "gaussian_chain")
    if frame_lambda_smooth is not None:
        if frame_lambda_smooth < 0:
            raise ValueError("frame_lambda_smooth must be non-negative")
        config["frame_pgm_smoother"]["lambda_smooth"] = float(frame_lambda_smooth)
        config["frame_pgm_smoother"].setdefault("type", "gaussian_chain")
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
    if early_stop_patience is not None:
        config["training"]["early_stop_patience"] = int(early_stop_patience)
    if progress_every is not None:
        config["training"]["progress_every"] = int(progress_every)
    if seed is not None:
        config["training"]["seed"] = int(seed)

    config["dataset"]["train_file"] = str(train_file)
    config["dataset"]["val_file"] = str(val_file)
    if run_dir is not None:
        config["output"]["run_dir"] = str(run_dir)
    return config


def subset_payload(payload: dict[str, Any], count: int) -> dict[str, Any]:
    if "dataset" in payload:
        count = min(count, len(payload["dataset"]))
        indices = list(range(count))
        return {
            **payload,
            "dataset": Subset(payload["dataset"], indices),
            "X_shape": (count, *payload["X_shape"][1:]),
            "labels": payload["labels"][:count].clone(),
            "metadata": payload.get("metadata", [])[:count],
        }
    return {
        "X": payload["X"][:count].clone(),
        "labels": payload["labels"][:count].clone(),
        "label_names": payload.get("label_names", []),
        "metadata": payload.get("metadata", [])[:count],
    }


def payload_shape(payload: dict[str, Any]) -> tuple[int, ...]:
    if "X_shape" in payload:
        return tuple(int(v) for v in payload["X_shape"])
    return tuple(int(v) for v in payload["X"].shape)


def payload_ndim(payload: dict[str, Any]) -> int:
    return len(payload_shape(payload))


def make_loader(
    payload: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = payload.get("dataset")
    if dataset is None:
        dataset = TensorDataset(payload["X"], payload["labels"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    epoch: int | None = None,
    epochs: int | None = None,
    phase: str = "train",
    progress_every: int = 0,
    overall_start_time: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    diagnostic_totals = {
        "correction_magnitude": 0.0,
        "observation_residual": 0.0,
        "smoothness_energy": 0.0,
    }
    frame_diagnostic_totals = {
        "frame_pgm_correction_magnitude": 0.0,
        "frame_pgm_observation_residual": 0.0,
        "frame_pgm_smoothness_energy": 0.0,
    }
    diagnostic_count = 0
    frame_diagnostic_count = 0
    phase_start_time = time.perf_counter()
    num_batches = len(loader)

    for batch_index, (X, labels) in enumerate(loader, start=1):
        X = X.to(device).float()
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if training:
                    logits = model(X)
                    debug = None
                else:
                    debug = model(X, return_debug=True)
                    logits = debug["logits"]
                loss = F.cross_entropy(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += batch_size
        if not training and isinstance(debug, dict):
            diagnostics = debug.get("pgm_diagnostics", {})
            if diagnostics:
                for name in diagnostic_totals:
                    value = diagnostics.get(name)
                    if value is not None:
                        diagnostic_totals[name] += float(value.detach().cpu()) * batch_size
                diagnostic_count += batch_size
            frame_diagnostics = debug.get("frame_pgm_diagnostics", {})
            if frame_diagnostics:
                frame_name_map = {
                    "correction_magnitude": "frame_pgm_correction_magnitude",
                    "observation_residual": "frame_pgm_observation_residual",
                    "smoothness_energy": "frame_pgm_smoothness_energy",
                }
                for source_name, target_name in frame_name_map.items():
                    value = frame_diagnostics.get(source_name)
                    if value is not None:
                        frame_diagnostic_totals[target_name] += float(value.detach().cpu()) * batch_size
                frame_diagnostic_count += batch_size
        if progress_every > 0 and (
            batch_index == 1 or batch_index % progress_every == 0 or batch_index == num_batches
        ):
            phase_elapsed = time.perf_counter() - phase_start_time
            phase_rate = batch_index / max(phase_elapsed, 1e-9)
            phase_eta = (num_batches - batch_index) / max(phase_rate, 1e-9)
            message = (
                f"  {phase}: batch {batch_index}/{num_batches} "
                f"({100.0 * batch_index / max(num_batches, 1):.1f}%) "
                f"elapsed={format_duration(phase_elapsed)} "
                f"phase_eta={format_duration(phase_eta)}"
            )
            if epoch is not None and epochs is not None and overall_start_time is not None:
                phase_weight = 0.85 if training else 0.15
                phase_base = 0.0 if training else 0.85
                epoch_fraction = phase_base + phase_weight * (batch_index / max(num_batches, 1))
                completed = (epoch - 1 + epoch_fraction) / max(epochs, 1)
                total_elapsed = time.perf_counter() - overall_start_time
                total_eta = total_elapsed * (1.0 - completed) / max(completed, 1e-9)
                message += f" total={100.0 * completed:.1f}% total_eta={format_duration(total_eta)}"
            print(message, flush=True)

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }
    if diagnostic_count:
        metrics.update(
            {
                name: total / diagnostic_count
                for name, total in diagnostic_totals.items()
            }
        )
    if frame_diagnostic_count:
        metrics.update(
            {
                name: total / frame_diagnostic_count
                for name, total in frame_diagnostic_totals.items()
            }
        )
    return metrics


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
    diff_config = config.get("diff_nn", {})
    diff_type = diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net"))
    model_config = config.get("model", {})
    model_type = str(model_config.get("name", model_config.get("type", "embedding_difference_pgm")))
    if model_type in {"p_series_trajectory_matrix", "p_series_trajectory"}:
        valid_diff_types = {"projected_pairwise_diff_net", "pairwise_diff_net"}
    else:
        valid_diff_types = {"pairwise_diff_net", "simple_concat_pairwise"}
    if diff_type not in valid_diff_types:
        raise NotImplementedError(
            f"diff_nn.diff_net_type must be one of {sorted(valid_diff_types)} for model.type={model_type!r}"
        )

    seed = int(training_config.get("seed", 0))
    set_reproducible_seed(seed)

    feature_layer = config.get("dataset", {}).get("feature_layer", config.get("backbone", {}).get("feature_layer"))
    if feature_layer is not None:
        feature_layer = str(feature_layer)

    train_payload = load_embedding_payload(train_file, feature_layer=feature_layer, split_name="train")
    val_payload = load_embedding_payload(val_file, feature_layer=feature_layer, split_name="test")
    if overfit_samples is not None:
        train_payload = subset_payload(train_payload, overfit_samples)
        val_payload = subset_payload(train_payload, overfit_samples)

    train_shape = payload_shape(train_payload)
    val_shape = payload_shape(val_payload)
    D = int(train_shape[-1])
    config.setdefault("backbone", {})["input_dim"] = D
    payload_backbone = train_payload.get("backbone_name")
    payload_format = train_payload.get("feature_format")
    if isinstance(payload_backbone, str) and payload_backbone.startswith("resnet50"):
        if payload_format == "spatial_map":
            selected_layer = train_payload.get("selected_feature_layer")
            if selected_layer == "layer3":
                config["backbone"]["name"] = "precomputed_resnet50_layer3"
            else:
                config["backbone"]["name"] = "precomputed_resnet50_layer4"
            config["backbone"]["feature_format"] = "spatial_map"
            config["backbone"]["spatial_tokens"] = int(train_shape[2])
        else:
            config["backbone"]["name"] = "precomputed_resnet50"
            config["backbone"]["feature_format"] = "vector"
    elif payload_ndim(train_payload) == 4:
        config["backbone"]["feature_format"] = "spatial_map"
        config["backbone"]["spatial_tokens"] = int(train_shape[2])
    if val_shape[-1] != D:
        raise ValueError(f"train D={D} but val D={val_shape[-1]}")
    if len(train_shape) != len(val_shape):
        raise ValueError(f"train X ndim={len(train_shape)} but val X ndim={len(val_shape)}")
    if len(train_shape) == 4 and train_shape[2] != val_shape[2]:
        raise ValueError(f"train spatial tokens={train_shape[2]} but val has {val_shape[2]}")

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
        train_payload,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        val_payload,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    if model_type in {"p_series_trajectory_matrix", "p_series_trajectory"}:
        model = PSeriesTrajectoryMatrixModel.from_config(config).to(device)
    else:
        model = EmbeddingDifferencePGMModel.from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training_config.get("amp", device.type == "cuda")) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    epochs = int(training_config.get("epochs", 30))
    save_best = bool(training_config.get("save_best", True))
    patience_config = training_config.get("early_stop_patience")
    early_stop_patience = None if patience_config is None else int(patience_config)
    if early_stop_patience is not None and early_stop_patience <= 0:
        early_stop_patience = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stopped = False
    stop_reason = "completed max epochs"
    history: list[dict[str, Any]] = []
    progress_every = int(training_config.get("progress_every", 0) or 0)
    overall_start_time = time.perf_counter()

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
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            epochs=epochs,
            phase="train",
            progress_every=progress_every,
            overall_start_time=overall_start_time,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                use_amp=use_amp,
                epoch=epoch,
                epochs=epochs,
                phase="val",
                progress_every=progress_every,
                overall_start_time=overall_start_time,
            )
        epoch_time_sec = time.perf_counter() - start_time
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["acc"],
            "val_correction_magnitude": val_metrics.get("correction_magnitude"),
            "val_observation_residual": val_metrics.get("observation_residual"),
            "val_smoothness_energy": val_metrics.get("smoothness_energy"),
            "val_frame_pgm_correction_magnitude": val_metrics.get("frame_pgm_correction_magnitude"),
            "val_frame_pgm_observation_residual": val_metrics.get("frame_pgm_observation_residual"),
            "val_frame_pgm_smoothness_energy": val_metrics.get("frame_pgm_smoothness_energy"),
            "lr": lr,
            "epoch_time_sec": epoch_time_sec,
        }
        history.append(row)
        append_train_log(run_paths.run_dir, row)
        diag_text = ""
        if "correction_magnitude" in val_metrics:
            diag_text = (
                f"pgm_corr={val_metrics['correction_magnitude']:.4f} "
                f"pgm_resid={val_metrics['observation_residual']:.4f} "
                f"pgm_smooth={val_metrics['smoothness_energy']:.4f} "
            )
        if "frame_pgm_correction_magnitude" in val_metrics:
            diag_text += (
                f"frame_pgm_corr={val_metrics['frame_pgm_correction_magnitude']:.4f} "
                f"frame_pgm_resid={val_metrics['frame_pgm_observation_residual']:.4f} "
                f"frame_pgm_smooth={val_metrics['frame_pgm_smoothness_energy']:.4f} "
            )
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f} "
            f"{diag_text}"
            f"progress={100.0 * epoch / max(epochs, 1):.1f}% "
            f"elapsed={format_duration(time.perf_counter() - overall_start_time)} "
            f"eta={format_duration((time.perf_counter() - overall_start_time) * (epochs - epoch) / max(epoch, 1))}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
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
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                        "config": config,
                        "epoch": epoch,
                        "best_val_top1": best_val_acc,
                        "best_val_loss": best_val_loss,
                    },
                    run_paths.best_checkpoint_path,
                )
        epochs_since_best = epoch - best_epoch
        if wandb_run is not None:
            log_row = {
                "train/loss": train_metrics["loss"],
                "val/loss": val_metrics["loss"],
                "train/top1": train_metrics["acc"],
                "val/top1": val_metrics["acc"],
                "lr": lr,
                "epoch": epoch,
            }
            if "correction_magnitude" in val_metrics:
                log_row.update(
                    {
                        "val/pgm_correction_magnitude": val_metrics["correction_magnitude"],
                        "val/pgm_observation_residual": val_metrics["observation_residual"],
                        "val/pgm_smoothness_energy": val_metrics["smoothness_energy"],
                    }
                )
            if "frame_pgm_correction_magnitude" in val_metrics:
                log_row.update(
                    {
                        "val/frame_pgm_correction_magnitude": val_metrics["frame_pgm_correction_magnitude"],
                        "val/frame_pgm_observation_residual": val_metrics["frame_pgm_observation_residual"],
                        "val/frame_pgm_smoothness_energy": val_metrics["frame_pgm_smoothness_energy"],
                    }
                )
            wandb_run.log(log_row)
        if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
            early_stopped = True
            stop_reason = f"no validation improvement for {early_stop_patience} epochs"
            print(
                f"early stopping at epoch {epoch:03d}: "
                f"best_val_top1={best_val_acc:.4f} at epoch {best_epoch:03d}; "
                f"{stop_reason}"
            )
            break

    final = history[-1]
    ablation_id = get_ablation_id(config)
    model_variant = get_model_variant(config)
    pgm_type = config.get("pgm_smoother", {}).get("type")
    frame_pgm_type = config.get("frame_pgm_smoother", {}).get("type", "none")
    lambda_smooth = None if pgm_type == "none" else config.get("pgm_smoother", {}).get("lambda_smooth")
    frame_lambda_smooth = (
        None if frame_pgm_type == "none" else config.get("frame_pgm_smoother", {}).get("lambda_smooth")
    )
    model_config = config.get("model", {})
    model_type = str(model_config.get("name", model_config.get("type", "embedding_difference_pgm")))
    if model_type in {"p_series_trajectory_matrix", "p_series_trajectory"}:
        pre_pgm_config = config.get("pre_pgm", {})
        post_pgm_config = config.get("pgm", {})
        pre_lambda = float(pre_pgm_config.get("lambda_frame", model_config.get("pre_lambda_frame", 0.0)) or 0.0)
        post_lambda = float(post_pgm_config.get("lambda_frame", model_config.get("lambda_frame", 0.0)) or 0.0)
        if bool(model_config.get("use_pre_pgm", False)) and pre_lambda > 0.0:
            frame_pgm_type = "pre_gaussian_chain"
            frame_lambda_smooth = pre_lambda
            lambda_smooth = pre_lambda
        elif bool(model_config.get("use_pgm", False)) and post_lambda > 0.0:
            frame_pgm_type = "post_projected_gaussian_chain"
            frame_lambda_smooth = post_lambda
            lambda_smooth = post_lambda
    info_enabled = bool(config.get("information_matrix", {}).get("enabled", ablation_id in {"E1.5", "E3"}))
    use_alpha = bool(config.get("information_matrix", {}).get("use_alpha", False)) if info_enabled else False
    diff_config = config.get("diff_nn", {})
    diff_net_type = diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net"))
    metrics = {
        "run_name": run_paths.run_name,
        "date": now_timestamp(),
        "ablation_id": ablation_id,
        "model_variant": model_variant,
        "diff_net_type": diff_net_type,
        "best_val_top1": best_val_acc,
        "best_val_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "stop_reason": stop_reason,
        "final_train_top1": final["train_top1"],
        "final_train_loss": final["train_loss"],
        "final_val_top1": final["val_top1"],
        "final_val_loss": final["val_loss"],
        "final_val_correction_magnitude": final.get("val_correction_magnitude"),
        "final_val_observation_residual": final.get("val_observation_residual"),
        "final_val_smoothness_energy": final.get("val_smoothness_energy"),
        "final_val_frame_pgm_correction_magnitude": final.get("val_frame_pgm_correction_magnitude"),
        "final_val_frame_pgm_observation_residual": final.get("val_frame_pgm_observation_residual"),
        "final_val_frame_pgm_smoothness_energy": final.get("val_frame_pgm_smoothness_energy"),
        "test_loss": None,
        "test_top1": None,
        "lambda_smooth": lambda_smooth,
        "frame_pgm_type": frame_pgm_type,
        "frame_lambda_smooth": frame_lambda_smooth,
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
