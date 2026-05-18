"""Generate a Manim-based synthetic temporal warm-up dataset.

Run from the project root:

    python scripts/generate_warmup_motion_manim.py --preview true
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "warmup_motion_manim" / "raw"

LABEL_NAMES = [
    "clockwise",
    "counter_clockwise",
    "horizontal_oscillation",
    "vertical_oscillation",
    "stationary",
]
FRAME_WIDTH = 8.0
FRAME_HEIGHT = 8.0
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate warmup_motion_manim raw frames.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_val", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=str_to_bool, default=False)
    return parser.parse_args()


def require_manim():
    try:
        import manim as mn  # type: ignore
        from manim.camera.camera import Camera  # type: ignore
    except ImportError as exc:
        print("pip install manim")
        raise SystemExit(1) from exc
    return mn, Camera


def rgb_to_hex(color: list[int] | tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(channel) for channel in color])


def pixel_to_scene(x_px: float, y_px: float, image_size: int) -> tuple[float, float, float]:
    x_scene = (x_px / image_size - 0.5) * FRAME_WIDTH
    y_scene = (0.5 - y_px / image_size) * FRAME_HEIGHT
    return x_scene, y_scene, 0.0


def pixels_to_scene_units(value_px: float, image_size: int) -> float:
    return value_px / image_size * FRAME_WIDTH


class ManimFrameRenderer:
    """Render one geometric object per frame using Manim's Cairo camera."""

    def __init__(self, mn: Any, camera_cls: Any, image_size: int) -> None:
        self.mn = mn
        self.camera_cls = camera_cls
        self.image_size = image_size
        self.configure_manim()

    def configure_manim(self) -> None:
        self.mn.config.pixel_width = self.image_size
        self.mn.config.pixel_height = self.image_size
        self.mn.config.frame_width = FRAME_WIDTH
        self.mn.config.frame_height = FRAME_HEIGHT
        self.mn.config.background_opacity = 1
        self.mn.config.disable_caching = True
        self.mn.config.renderer = "cairo"
        self.mn.config.verbosity = "ERROR"

    def make_mobject(
        self,
        object_type: str,
        color: list[int],
        object_size_px: float,
        position_px: tuple[float, float],
    ) -> Any:
        size_units = pixels_to_scene_units(object_size_px, self.image_size)
        fill_color = rgb_to_hex(color)

        if object_type == "circle":
            mobject = self.mn.Circle(
                radius=size_units,
                fill_color=fill_color,
                fill_opacity=1.0,
                stroke_width=0,
            )
        elif object_type == "square":
            mobject = self.mn.Square(
                side_length=2.0 * size_units,
                fill_color=fill_color,
                fill_opacity=1.0,
                stroke_width=0,
            )
        elif object_type == "triangle":
            mobject = self.mn.RegularPolygon(
                n=3,
                radius=size_units * 1.25,
                fill_color=fill_color,
                fill_opacity=1.0,
                stroke_width=0,
            )
        else:
            raise ValueError(f"unknown object_type: {object_type}")

        mobject.move_to(pixel_to_scene(position_px[0], position_px[1], self.image_size))
        return mobject

    def render(
        self,
        object_type: str,
        color: list[int],
        background: list[int],
        object_size_px: float,
        position_px: tuple[float, float],
    ) -> Image.Image:
        camera = self.camera_cls(background_color=rgb_to_hex(background))
        mobject = self.make_mobject(object_type, color, object_size_px, position_px)
        camera.capture_mobjects([mobject])

        if hasattr(camera, "get_image"):
            image = camera.get_image()
        else:
            image = Image.fromarray(camera.pixel_array)
        return image.convert("RGB")


def balanced_labels(num_samples: int, rng: random.Random) -> list[int]:
    labels = [idx % len(LABEL_NAMES) for idx in range(num_samples)]
    rng.shuffle(labels)
    return labels


def sample_object_color(rng: random.Random, background: list[int]) -> list[int]:
    background_mean = sum(background) / 3.0
    for _ in range(100):
        color = [rng.randint(35, 230) for _ in range(3)]
        if abs(sum(color) / 3.0 - background_mean) >= 45:
            return color
    return [40, 95, 220]


def sample_background(rng: random.Random) -> list[int]:
    base = rng.randint(225, 248)
    tint = [rng.randint(-5, 5) for _ in range(3)]
    return [max(0, min(255, base + delta)) for delta in tint]


def sample_visual_params(rng: random.Random) -> dict[str, Any]:
    background = sample_background(rng)
    return {
        "object_type": rng.choice(["circle", "square", "triangle"]),
        "color": sample_object_color(rng, background),
        "object_size": rng.randint(12, 22),
        "background": background,
    }


def sample_extent(rng: random.Random, image_size: int, object_size: int) -> float:
    max_extent = image_size / 2.0 - object_size - 12.0
    high = min(68.0, max_extent)
    low = min(42.0, high * 0.75)
    if high <= 8:
        raise ValueError("image_size is too small for controlled motion")
    return rng.uniform(low, high)


def sample_motion_params(
    label: int,
    rng: random.Random,
    image_size: int,
    object_size: int,
) -> dict[str, Any]:
    if label in {0, 1}:
        radius = sample_extent(rng, image_size, object_size)
        margin = radius + object_size + 6.0
        return {
            "start_angle": rng.uniform(0.0, 2.0 * math.pi),
            "speed": rng.uniform(0.26, 0.48),
            "center": [
                rng.uniform(margin, image_size - margin),
                rng.uniform(margin, image_size - margin),
            ],
            "radius": radius,
        }

    if label == 2:
        amplitude = sample_extent(rng, image_size, object_size)
        x_margin = amplitude + object_size + 6.0
        y_margin = object_size + 12.0
        return {
            "phase": rng.uniform(0.0, 2.0 * math.pi),
            "speed": rng.uniform(0.34, 0.62),
            "center": [
                rng.uniform(x_margin, image_size - x_margin),
                rng.uniform(y_margin, image_size - y_margin),
            ],
            "amplitude": amplitude,
        }

    if label == 3:
        amplitude = sample_extent(rng, image_size, object_size)
        x_margin = object_size + 12.0
        y_margin = amplitude + object_size + 6.0
        return {
            "phase": rng.uniform(0.0, 2.0 * math.pi),
            "speed": rng.uniform(0.34, 0.62),
            "center": [
                rng.uniform(x_margin, image_size - x_margin),
                rng.uniform(y_margin, image_size - y_margin),
            ],
            "amplitude": amplitude,
        }

    margin = object_size + 12.0
    return {
        "speed": 0.0,
        "center": [
            rng.uniform(margin, image_size - margin),
            rng.uniform(margin, image_size - margin),
        ],
        "jitter": 0.35,
    }


def position_at_t(label: int, params: dict[str, Any], t: int, rng: random.Random) -> tuple[float, float]:
    center_x, center_y = params["center"]

    if label == 0:
        angle = params["start_angle"] - params["speed"] * t
        return (
            center_x + params["radius"] * math.cos(angle),
            center_y + params["radius"] * math.sin(angle),
        )
    if label == 1:
        angle = params["start_angle"] + params["speed"] * t
        return (
            center_x + params["radius"] * math.cos(angle),
            center_y + params["radius"] * math.sin(angle),
        )
    if label == 2:
        phase = params["phase"] + params["speed"] * t
        return center_x + params["amplitude"] * math.sin(phase), center_y
    if label == 3:
        phase = params["phase"] + params["speed"] * t
        return center_x, center_y + params["amplitude"] * math.sin(phase)

    jitter = params["jitter"]
    return center_x + rng.uniform(-jitter, jitter), center_y + rng.uniform(-jitter, jitter)


def generate_sample(
    renderer: ManimFrameRenderer,
    split_dir: Path,
    split: str,
    sample_idx: int,
    label: int,
    T: int,
    image_size: int,
    rng: random.Random,
    raw_dir: Path,
) -> dict[str, Any]:
    sample_id = f"sample_{sample_idx:06d}"
    sample_dir = split_dir / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)

    visual_params = sample_visual_params(rng)
    motion_params = sample_motion_params(
        label=label,
        rng=rng,
        image_size=image_size,
        object_size=visual_params["object_size"],
    )
    params = {**visual_params, **motion_params}

    frame_paths: list[str] = []
    for t in range(T):
        position = position_at_t(label, params, t, rng)
        frame = renderer.render(
            object_type=params["object_type"],
            color=params["color"],
            background=params["background"],
            object_size_px=params["object_size"],
            position_px=position,
        )
        frame_path = sample_dir / f"frame_{t:03d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path.relative_to(raw_dir).as_posix())

    return {
        "sample_id": sample_id,
        "split": split,
        "label": label,
        "label_name": LABEL_NAMES[label],
        "T": T,
        "frame_paths": frame_paths,
        "params": params,
    }


def generate_split(
    renderer: ManimFrameRenderer,
    split: str,
    num_samples: int,
    raw_dir: Path,
    T: int,
    image_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    split_dir = raw_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for sample_idx, label in enumerate(balanced_labels(num_samples, rng)):
        records.append(
            generate_sample(
                renderer=renderer,
                split_dir=split_dir,
                split=split,
                sample_idx=sample_idx,
                label=label,
                T=T,
                image_size=image_size,
                rng=rng,
                raw_dir=raw_dir,
            )
        )

    metadata_path = raw_dir / f"metadata_{split}.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def write_preview_grid(
    metadata: dict[str, Any],
    raw_dir: Path,
    preview_dir: Path,
    thumb_size: int = 96,
) -> None:
    frame_paths = [raw_dir / frame_path for frame_path in metadata["frame_paths"]]
    cols = math.ceil(math.sqrt(len(frame_paths)))
    rows = math.ceil(len(frame_paths) / cols)
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (255, 255, 255))

    for idx, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as frame:
            thumb = frame.convert("RGB").resize((thumb_size, thumb_size), RESAMPLE_BICUBIC)
        grid.paste(thumb, ((idx % cols) * thumb_size, (idx // cols) * thumb_size))

    preview_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{metadata['split']}_{metadata['sample_id']}_{metadata['label_name']}.png"
    grid.save(preview_dir / filename)


def generate_previews(
    all_metadata: dict[str, list[dict[str, Any]]],
    raw_dir: Path,
    preview_dir: Path,
) -> None:
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for split, records in all_metadata.items():
        seen_labels: set[int] = set()
        for record in records:
            label = int(record["label"])
            if label in seen_labels:
                continue
            write_preview_grid(record, raw_dir=raw_dir, preview_dir=preview_dir)
            seen_labels.add(label)
            if len(seen_labels) == len(LABEL_NAMES):
                break
        print(f"Selected {len(seen_labels)} {split} preview samples.")


def main() -> None:
    args = parse_args()
    if args.T < 2:
        raise SystemExit("T must be at least 2 for temporal motion.")
    if args.image_size < 128:
        raise SystemExit("image_size must be at least 128.")

    mn, camera_cls = require_manim()
    renderer = ManimFrameRenderer(mn, camera_cls, args.image_size)
    rng = random.Random(args.seed)

    raw_dir = args.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {
        "train": args.num_train,
        "val": args.num_val,
        "test": args.num_test,
    }

    all_metadata = {
        split: generate_split(
            renderer=renderer,
            split=split,
            num_samples=num_samples,
            raw_dir=raw_dir,
            T=args.T,
            image_size=args.image_size,
            rng=rng,
        )
        for split, num_samples in split_sizes.items()
    }

    if args.preview:
        generate_previews(all_metadata, raw_dir=raw_dir, preview_dir=raw_dir.parent / "preview")

    print(f"Generated warmup_motion_manim dataset under {raw_dir}")
    for split, records in all_metadata.items():
        print(f"  {split}: {len(records)} samples")


if __name__ == "__main__":
    main()
