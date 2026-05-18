"""Extract CLIP ViT-B/16 frame embeddings for the warm-up motion dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "warmup_motion_manim" / "raw"
DEFAULT_EMB_DIR = PROJECT_ROOT / "data" / "warmup_motion_manim" / "embeddings"
SPLITS = ["train", "val", "test"]
LABEL_NAMES = [
    "clockwise",
    "counter_clockwise",
    "horizontal_oscillation",
    "vertical_oscillation",
    "stationary",
]
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for warm-up motion.")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--backbone_name", default="ViT-B/16")
    parser.add_argument("--embedding_subdir", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available.")
    return torch.device(device_name)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of metadata records")
    return data


def preprocess_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE):
            image = image.resize((CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), RESAMPLE_BICUBIC)
        image_bytes = image.tobytes()

    tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).clone()
    tensor = tensor.view(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0)
    return (tensor - CLIP_MEAN) / CLIP_STD


def load_encoder(
    device: torch.device,
    backbone_name: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    try:
        from models import CLIPFrameEncoder

        encoder = CLIPFrameEncoder(
            backbone_name=backbone_name,
            freeze=True,
            normalize=True,
            device=device,
        )
        encoder.eval()

        def encode_with_project_encoder(frames: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return encoder(frames.unsqueeze(1)).squeeze(1).detach().cpu()

        return encode_with_project_encoder, f"models.CLIPFrameEncoder({backbone_name})"
    except ImportError as exc:
        if "OpenAI CLIP is required" in str(exc) or "No module named 'clip'" in str(exc):
            raise

    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required for embedding extraction.\n"
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, _ = clip.load(backbone_name, device=device, jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def encode_direct(frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = model.encode_image(frames.to(device))
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            return embeddings.detach().cpu()

    return encode_direct, f"direct OpenAI CLIP({backbone_name})"


def encode_frame_paths(
    frame_paths: list[Path],
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []

    for idx, frame_path in enumerate(frame_paths, start=1):
        batch.append(preprocess_image(frame_path))
        if len(batch) == batch_size or idx == len(frame_paths):
            frames = torch.stack(batch, dim=0)
            chunks.append(encode_batch(frames))
            print(f"    encoded {idx}/{len(frame_paths)} frames")
            batch.clear()

    return torch.cat(chunks, dim=0)


def process_split(
    split: str,
    raw_dir: Path,
    out_dir: Path,
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> None:
    metadata_path = raw_dir / f"metadata_{split}.json"
    split_metadata = load_metadata(metadata_path)
    if not split_metadata:
        raise ValueError(f"{metadata_path} contains no samples")

    lengths = {int(record["T"]) for record in split_metadata}
    if len(lengths) != 1:
        raise ValueError(f"{split} contains mixed sequence lengths: {sorted(lengths)}")
    T = lengths.pop()

    frame_paths: list[Path] = []
    labels: list[int] = []
    for record in split_metadata:
        labels.append(int(record["label"]))
        for frame_rel in record["frame_paths"]:
            frame_path = raw_dir / frame_rel
            if not frame_path.is_file():
                raise FileNotFoundError(f"missing frame: {frame_path}")
            frame_paths.append(frame_path)

    print(f"\n{split}: {len(split_metadata)} videos, {len(frame_paths)} frames")
    flat_embeddings = encode_frame_paths(frame_paths, encode_batch, batch_size)
    if flat_embeddings.shape != (len(split_metadata) * T, 512):
        raise ValueError(f"expected flat embeddings [{len(split_metadata) * T}, 512], got {tuple(flat_embeddings.shape)}")

    X = flat_embeddings.reshape(len(split_metadata), T, 512).contiguous()
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    payload = {
        "X": X,
        "labels": labels_tensor,
        "label_names": LABEL_NAMES,
        "metadata": split_metadata,
    }
    out_path = out_dir / f"{split}.pt"
    torch.save(payload, out_path)
    print(f"  saved {out_path} with X={tuple(X.shape)}, labels={tuple(labels_tensor.shape)}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch_size must be positive.")

    device = resolve_device(args.device)
    out_dir = args.out_dir / args.embedding_subdir if args.embedding_subdir else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        encode_batch, encoder_name = load_encoder(device, args.backbone_name)
    except ImportError as exc:
        print(exc)
        print("pip install git+https://github.com/openai/CLIP.git")
        raise SystemExit(1) from exc

    print(f"Using {encoder_name} on {device}.")
    for split in SPLITS:
        process_split(
            split=split,
            raw_dir=args.raw_dir,
            out_dir=out_dir,
            encode_batch=encode_batch,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
