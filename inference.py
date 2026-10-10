#!/usr/bin/env python3
"""Unified DuS-DiFuse inference entry point."""

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dus_difuse.dataset import FusionInferenceDataset
from dus_difuse.model import DetailRestorationFidelityModule, Diffusion, DuSDiFuseModel
from dus_difuse.pipeline import DuSDiFusePipeline
from dus_difuse.utils.common import instantiate_from_config


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(value: Optional[str], label: str) -> Path:
    path = project_path(value)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_state(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("state_dict", state) if isinstance(state, dict) else state


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mask(
    path: Path,
    original_height: int,
    original_width: int,
    padded_height: int,
    padded_width: int,
    device: str,
) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    mask = cv2.resize(
        mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )
    mask = np.pad(
        mask,
        ((0, padded_height - original_height), (0, padded_width - original_width)),
        mode="constant",
    )
    return torch.from_numpy(mask >= 128).float().unsqueeze(0).to(device)


def resolve_mask_path(mask_arg: str, image_name: str) -> Path:
    path = project_path(mask_arg)
    if path.is_dir():
        candidate = path / image_name
        if not candidate.is_file():
            candidate = path / f"{Path(image_name).stem}.png"
        return candidate
    return path


def save_attention_map(mask: torch.Tensor, path: Path, height: int, width: int) -> None:
    """Save a segmentation mask as a color attention heatmap."""
    attention = mask[:height, :width].detach().float().clamp(0, 1).cpu().numpy()
    attention = np.rint(attention * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), heatmap):
        raise OSError(f"Failed to save attention map: {path}")


def build_generation_mask(args, batch, height: int, width: int, segmenter):
    if not args.generation:
        binary = torch.ones((1, height, width), device=args.device)
        mode = "off"
    elif args.mask:
        mask_path = resolve_mask_path(args.mask, batch["name"][0])
        original_h = int(batch["original_size"][0, 0])
        original_w = int(batch["original_size"][0, 1])
        binary = load_mask(
            mask_path, original_h, original_w, height, width, args.device
        )
        mode = "mask"
    elif args.text_prompt:
        vi = (batch["visible"][0].to(args.device) + 1) / 2
        ir = (batch["infrared"][0].to(args.device) + 1) / 2
        binary = torch.maximum(
            segmenter.mask(vi, args.text_prompt, args.box_threshold, args.text_threshold),
            segmenter.mask(ir, args.text_prompt, args.box_threshold, args.text_threshold),
        )
        mode = "text"
    else:
        binary = torch.ones((1, height, width), device=args.device)
        mode = "global"
    condition = binary.repeat(3, 1, 1).unsqueeze(0) * 2 - 1
    return condition, binary, mode


def create_segmenter(args):
    if not (args.generation and args.text_prompt and not args.mask):
        return None
    from dus_difuse.target_att import TargetSegmenter

    return TargetSegmenter(
        str(require_file(args.grounding_config, "GroundingDINO config")),
        str(require_file(args.grounding_checkpoint, "GroundingDINO checkpoint")),
        str(require_file(args.sam_checkpoint, "SAM checkpoint")),
        str(project_path(args.bert_path)),
        args.device,
        args.sam_type,
    )


def main(args: argparse.Namespace) -> None:
    set_reproducible_seed(args.seed)
    config_path = require_file(args.config, "configuration")
    cfg = OmegaConf.load(config_path)
    cfg.model.dus_difuse.params.clip_cfg.pretrained_path = str(
        require_file(
            cfg.model.dus_difuse.params.clip_cfg.pretrained_path,
            "MDA checkpoint",
        )
    )
    if not args.generation:
        cfg.model.dus_difuse.params.controlnet_cfg = None
        cfg.model.dus_difuse.params.SD_unet_cfg = None
        cfg.model.dus_difuse.params.clip_text_cfg = None
    output_dir = project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = output_dir / "mask"

    model: DuSDiFuseModel = instantiate_from_config(cfg.model.dus_difuse)
    sd_path = require_file(cfg.checkpoints.sd, "Stable Diffusion checkpoint")
    if args.generation:
        unused, missing = model.load_pretrained_sd(load_state(sd_path))
    else:
        unused, missing = model.load_pretrained_VAE_only(load_state(sd_path))
    print(f"Loaded base checkpoint ({len(unused)} unused, {len(missing)} missing keys)")
    model.load_unet_from_ckpt(load_state(require_file(cfg.checkpoints.sde, "SDE checkpoint")))
    if args.fusion:
        model.load_gfcm_from_ckpt(
            load_state(require_file(cfg.checkpoints.fusion, "fusion checkpoint"))
        )
    if args.generation:
        model.load_controlnet_from_ckpt(
            load_state(require_file(cfg.checkpoints.generation, "generation checkpoint"))
        )

    drfm = None
    if args.vae_enhance:
        drfm = DetailRestorationFidelityModule(cfg.model.drfm.params)
        drfm.load_state_dict(
            load_state(require_file(cfg.checkpoints.vae_enhance, "VAE enhancement checkpoint")),
            strict=True,
        )
        drfm.eval().to(args.device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    model.eval().to(args.device)
    pipeline = DuSDiFusePipeline(model, diffusion, args.device)
    segmenter = create_segmenter(args)

    dataset = FusionInferenceDataset(
        root=str(project_path(args.input_root)) if args.input_root else None,
        visible_dir=str(project_path(args.visible_dir)) if args.visible_dir else None,
        infrared_dir=str(project_path(args.infrared_dir)) if args.infrared_dir else None,
        divisor=args.divisor,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    print(
        f"Processing {len(dataset)} pair(s) with {args.steps} diffusion steps; "
        f"results: {output_dir}"
    )

    for batch in loader:
        vi = batch["visible"].to(args.device)
        ir = batch["infrared"].to(args.device)
        height, width = vi.shape[-2:]
        condition, binary_mask, mode = build_generation_mask(
            args, batch, height, width, segmenter
        )
        original_h = int(batch["original_size"][0, 0])
        original_w = int(batch["original_size"][0, 1])
        mask_coverage = binary_mask[:, :original_h, :original_w].float().mean(
            dim=(-2, -1)
        )
        started = time.perf_counter()
        result, _ = pipeline.run(
            visible=vi,
            infrared=ir,
            steps=args.steps,
            visible_clip=batch["visible_clip"].to(args.device),
            infrared_clip=batch["infrared_clip"].to(args.device),
            generation_mask=condition,
            generation_mask_coverage=mask_coverage,
            prompt=args.text_prompt if args.generation else "",
            cfg_scale=args.cfg_scale,
            start_point=args.start_point,
            fusion=args.fusion,
            generation=args.generation,
            drfm=drfm,
            tiled=args.tiled,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
        )
        save_image(
            ((result[:, :, :original_h, :original_w] + 1) / 2).clamp(0, 1),
            output_dir / batch["name"][0],
        )
        if args.save_mask and args.generation:
            save_attention_map(
                binary_mask[0],
                mask_output_dir / batch["name"][0],
                original_h,
                original_w,
            )
        print(f"{batch['name'][0]}: generation={mode}, {time.perf_counter() - started:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input-root",
        default="data/example",
        help="Directory containing vi/ and ir/ or source_1/ and source_2/",
    )
    inputs.add_argument("--visible-dir", help="Visible image directory (also set --infrared-dir)")
    parser.add_argument("--infrared-dir", help="Infrared image directory")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Reverse diffusion steps (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--divisor", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--start-point", choices=("random", "cond"), default="random")
    parser.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-prompt", default="")
    parser.add_argument("--mask", help="Binary mask file or directory; overrides text segmentation")
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save generation masks as color attention heatmaps",
    )
    parser.add_argument("--grounding-config", default="weights/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--grounding-checkpoint", default="weights/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--sam-checkpoint", default="weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--bert-path", default="weights/bert-base-uncased")
    parser.add_argument("--sam-type", default="vit_h")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=384)
    args = parser.parse_args()
    if bool(args.visible_dir) != bool(args.infrared_dir):
        parser.error("--visible-dir and --infrared-dir must be provided together")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.tile_size <= 0 or args.tile_stride <= 0:
        parser.error("--tile-size and --tile-stride must be positive")
    if args.tiled and (args.tile_size % 64 or args.tile_stride % 64):
        parser.error("tiled inference requires tile size and stride divisible by 64")
    return args


if __name__ == "__main__":
    main(parse_args())
