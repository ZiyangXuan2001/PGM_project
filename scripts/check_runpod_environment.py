"""Check RunPod/Linux readiness without requiring real Diving48 data."""

from __future__ import annotations

import argparse
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EmbeddingDifferencePGMModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.experiment_manager import ABLATION_INFO  # noqa: E402
from utils.training import resolve_device, set_controlled_variant  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RunPod environment and model smoke forward.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def check_output_dir() -> Path:
    check_dir = PROJECT_ROOT / "outputs" / "runpod_env_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    probe = check_dir / "write_test.txt"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return check_dir


def run_model_checks(config: dict, device: torch.device) -> None:
    base = deepcopy(config)
    X = torch.randn(2, 16, 512, device=device)
    for ablation_id in ["E0", "E1", "E2", "E3", "E4"]:
        model_config = deepcopy(base)
        set_controlled_variant(model_config, ablation_id=ablation_id)
        model = EmbeddingDifferencePGMModel.from_config(model_config).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X)
        print(f"{ablation_id} {ABLATION_INFO[ablation_id]['variant']} logits.shape: {tuple(logits.shape)}")
        if tuple(logits.shape) != (2, 48):
            raise RuntimeError(f"{ablation_id} logits shape check failed: {tuple(logits.shape)}")


def main() -> None:
    args = parse_args()
    print("RunPod environment check")
    print_kv("Python", sys.version.replace("\n", " "))
    print_kv("Platform", platform.platform())
    print_kv("PyTorch", torch.__version__)
    print_kv("CUDA available", torch.cuda.is_available())
    print_kv("CUDA version", torch.version.cuda)
    if torch.cuda.is_available():
        print_kv("GPU count", torch.cuda.device_count())
        print_kv("GPU name", torch.cuda.get_device_name(0))
    else:
        print_kv("GPU name", "none")
    print_kv("Current working directory", Path.cwd())
    print_kv("/workspace exists", Path("/workspace").exists())

    config = load_config(str(args.config))
    print_kv("Config loaded", args.config)

    output_dir = check_output_dir()
    print_kv("Output directory writable", output_dir)

    device = resolve_device(args.device)
    print_kv("Selected device", device)
    run_model_checks(config, device)
    print("Environment check passed.")


if __name__ == "__main__":
    main()
