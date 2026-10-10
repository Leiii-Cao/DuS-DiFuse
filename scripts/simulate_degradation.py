#!/usr/bin/env python3
"""Create reproducible degraded visible/infrared test pairs.

Each input folder has its own list of allowed degradation types.  For every
image, exactly one type is sampled uniformly from the corresponding list.  A
single-item list therefore applies the same degradation type to the whole
folder.  No cross-modal swapping or DRFM generic degradation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dus_difuse.dataset.degradation_sys import (  # noqa: E402
    add_blur_to_vi_img,
    add_haze_to_vi_img,
    add_lowcontrast_to_ir_img,
    add_noise_to_ir_img,
    add_noise_to_vi_img,
    add_rain_to_vi_img,
    add_snow_to_vi_img,
    add_stripe_to_ir_image,
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VISIBLE_TYPES = ("none", "noise", "blur", "rain", "snow", "haze")
INFRARED_TYPES = ("none", "noise", "lowcontrast", "stripe")


def project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_input_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if args.input_root:
        root = project_path(args.input_root)
        assert root is not None
        for visible_name, infrared_name in (
            ("vis", "ir"),
            ("vi", "ir"),
            ("source_1", "source_2"),
        ):
            visible_dir = root / visible_name
            infrared_dir = root / infrared_name
            if visible_dir.is_dir() and infrared_dir.is_dir():
                depth_dir = project_path(args.depth_dir) if args.depth_dir else root / "depth"
                return visible_dir, infrared_dir, depth_dir if depth_dir.is_dir() else None
        raise FileNotFoundError(
            f"Expected vis/ir, vi/ir, or source_1/source_2 under input root: {root}"
        )

    visible_dir = project_path(args.visible_dir)
    infrared_dir = project_path(args.infrared_dir)
    if visible_dir is None or infrared_dir is None:
        raise ValueError("Provide --input-root or both --visible-dir and --infrared-dir")
    if not visible_dir.is_dir() or not infrared_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image directories: {visible_dir} and {infrared_dir}"
        )
    depth_dir = project_path(args.depth_dir)
    if depth_dir is not None and not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")
    return visible_dir, infrared_dir, depth_dir


def paired_names(visible_dir: Path, infrared_dir: Path) -> list[str]:
    visible = {
        path.name
        for path in visible_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    infrared = {
        path.name
        for path in infrared_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(visible & infrared)
    if not names:
        raise RuntimeError("No filename-aligned visible/infrared pairs were found")
    missing = visible ^ infrared
    if missing:
        print(f"Warning: ignoring {len(missing)} unpaired image(s)")
    return names


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    image = np.clip(image, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image: {path}")


def sample_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def apply_visible(
    image: np.ndarray,
    depth: np.ndarray | None,
    allowed_types: list[str],
) -> tuple[np.ndarray, str, float]:
    operations: dict[str, Callable[[np.ndarray], tuple[np.ndarray, float]]] = {
        "noise": lambda value: add_noise_to_vi_img(value, return_params=True),
        "blur": lambda value: add_blur_to_vi_img(value, return_params=True),
        "rain": lambda value: add_rain_to_vi_img(value, return_params=True),
        "snow": lambda value: add_snow_to_vi_img(value, return_params=True),
    }
    if depth is not None:
        operations["haze"] = lambda value: add_haze_to_vi_img(
            value, depth, return_params=True
        )
    selected = random.choice(allowed_types)
    if selected == "none":
        return image, "none", 0.0
    if selected == "haze" and depth is None:
        raise ValueError("Visible haze degradation requires a matching depth image")
    output, score = operations[selected](image)
    return output, selected, float(score)


def apply_infrared(
    image: np.ndarray,
    allowed_types: list[str],
) -> tuple[np.ndarray, str, float]:
    operations: dict[str, Callable[[np.ndarray], tuple[np.ndarray, float]]] = {
        "noise": lambda value: add_noise_to_ir_img(value, return_params=True),
        "lowcontrast": lambda value: add_lowcontrast_to_ir_img(value, return_params=True),
        "stripe": lambda value: add_stripe_to_ir_image(value, return_params=True),
    }
    selected = random.choice(allowed_types)
    if selected == "none":
        return image, "none", 0.0
    output, score = operations[selected](image)
    return output, selected, float(score)


def output_name(source_name: str) -> str:
    return f"{Path(source_name).stem}.png"


def directory_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.name != value:
        raise argparse.ArgumentTypeError("must be a single directory name")
    return value


def preflight_outputs(
    names: list[str],
    visible_output_dir: Path,
    infrared_output_dir: Path,
    output_root: Path,
    overwrite: bool,
) -> list[tuple[str, Path, Path]]:
    planned = [
        (
            name,
            visible_output_dir / output_name(name),
            infrared_output_dir / output_name(name),
        )
        for name in names
    ]
    output_names = [visible.name for _, visible, _ in planned]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError(
            "Input filenames with different extensions collapse to the same PNG filename"
        )
    if not overwrite:
        conflicts = [
            path for _, visible, infrared in planned for path in (visible, infrared) if path.exists()
        ]
        manifest = output_root / "degradation_manifest.jsonl"
        if manifest.exists():
            conflicts.append(manifest)
        if conflicts:
            raise FileExistsError(
                f"Output already exists (for example {conflicts[0]}); use --overwrite"
            )
    return planned


def simulate(args: argparse.Namespace) -> None:
    visible_dir, infrared_dir, depth_dir = resolve_input_dirs(args)
    output_root = project_path(args.output_root)
    assert output_root is not None
    visible_output_dir = (output_root / args.visible_output_name).resolve()
    infrared_output_dir = (output_root / args.infrared_output_name).resolve()
    if visible_output_dir == infrared_output_dir:
        raise ValueError("Visible and infrared output directories must be different")
    input_dirs = {visible_dir, infrared_dir}
    if depth_dir is not None:
        input_dirs.add(depth_dir)
    if visible_output_dir in input_dirs or infrared_output_dir in input_dirs:
        raise ValueError("Output directories must not overwrite input directories")

    names = paired_names(visible_dir, infrared_dir)
    if args.limit is not None:
        names = names[: args.limit]
    if depth_dir is not None and "haze" in args.visible_types:
        missing_depth = [name for name in names if not (depth_dir / name).is_file()]
        if missing_depth:
            raise FileNotFoundError(
                f"Missing {len(missing_depth)} filename-aligned depth image(s); "
                f"for example: {depth_dir / missing_depth[0]}"
            )
    elif "haze" in args.visible_types:
        raise ValueError("--visible-types haze requires --depth-dir or input-root/depth")
    planned = preflight_outputs(
        names,
        visible_output_dir,
        infrared_output_dir,
        output_root,
        args.overwrite,
    )

    visible_output_dir.mkdir(parents=True, exist_ok=True)
    infrared_output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, visible_output, infrared_output) in enumerate(planned, start=1):
        visible = read_rgb(visible_dir / name)
        infrared = read_rgb(infrared_dir / name)
        if visible.shape != infrared.shape:
            raise ValueError(f"Pair has different spatial sizes: {name}")

        depth_path = (
            depth_dir / name
            if depth_dir is not None and "haze" in args.visible_types
            else None
        )
        depth = read_rgb(depth_path) if depth_path is not None and depth_path.is_file() else None
        if depth is not None and depth.shape != visible.shape:
            raise ValueError(f"Depth/image sizes differ: {depth_path}")

        visible_seed = sample_seed(args.seed, f"visible:{name}")
        seed_all(visible_seed)
        visible_degraded, visible_type, visible_score = apply_visible(
            visible, depth, args.visible_types
        )
        infrared_seed = sample_seed(args.seed, f"infrared:{name}")
        seed_all(infrared_seed)
        infrared_degraded, infrared_type, infrared_score = apply_infrared(
            infrared, args.infrared_types
        )

        write_rgb_png(visible_output, visible_degraded)
        write_rgb_png(infrared_output, infrared_degraded)
        records.append(
            {
                "name": output_name(name),
                "source_name": name,
                "visible": {
                    "seed": visible_seed,
                    "type": visible_type,
                    "score": visible_score,
                    "path": str(visible_output),
                },
                "infrared": {
                    "seed": infrared_seed,
                    "type": infrared_type,
                    "score": infrared_score,
                    "path": str(infrared_output),
                },
            }
        )
        print(
            f"[{index}/{len(planned)}] {name}: "
            f"vi={visible_type}, ir={infrared_type}"
        )

    manifest_path = output_root / "degradation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} degraded pair(s) to {output_root}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input-root",
        help="Paired root containing vis/ir, vi/ir, or source_1/source_2",
    )
    inputs.add_argument("--visible-dir", help="Visible image directory")
    parser.add_argument("--infrared-dir", help="Infrared directory used with --visible-dir")
    parser.add_argument("--depth-dir", help="Optional filename-aligned depth directory for haze")
    parser.add_argument("--output-root", required=True, help="Parent output directory")
    parser.add_argument(
        "--visible-output-name",
        type=directory_name,
        default="vis_lq",
        help="Visible output directory name under output-root (default: vis_lq)",
    )
    parser.add_argument(
        "--infrared-output-name",
        type=directory_name,
        default="ir_lq",
        help="Infrared output directory name under output-root (default: ir_lq)",
    )
    parser.add_argument(
        "--visible-types",
        nargs="+",
        choices=VISIBLE_TYPES,
        default=["noise", "blur", "rain", "snow"],
        help="Allowed visible degradation types; one is sampled per image",
    )
    parser.add_argument(
        "--infrared-types",
        nargs="+",
        choices=INFRARED_TYPES,
        default=["noise", "lowcontrast", "stripe"],
        help="Allowed infrared degradation types; one is sampled per image",
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--limit", type=int, help="Process only the first N pairs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.visible_dir) != bool(args.infrared_dir):
        parser.error("--visible-dir and --infrared-dir must be provided together")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    simulate(parse_args())
