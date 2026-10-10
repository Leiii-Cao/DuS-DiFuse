"""Shared helpers for the public training entry points."""

from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def resolve_train_paths(cfg):
    """Resolve paths from a training YAML relative to the repository root."""
    for key in (
        "dataset_root", "sd_path", "exp_dir", "unetresume", "skipresume",
        "resume", "label_root", "motion_kernel_path",
    ):
        if key in cfg.train and cfg.train[key] is not None:
            cfg.train[key] = project_path(cfg.train[key])
    if "image_roots" in cfg.train:
        cfg.train.image_roots = [project_path(path) for path in cfg.train.image_roots]
    clip_cfg = cfg.model.dus_difuse.params.get("clip_cfg")
    if clip_cfg and clip_cfg.get("pretrained_path"):
        clip_cfg.pretrained_path = project_path(clip_cfg.pretrained_path)
    return cfg


def load_state(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def paired_random_crop(
    batch,
    crop_size=256,
    divisor=64,
    clip_size=224,
    raw=False,
    max_crop_size=None,
):
    """Crop paired tensors consistently and pad them to a VAE-safe size."""
    image_keys = ["vi_hq", "vi_lq", "ir_hq", "ir_lq"]
    if raw:
        image_keys = ["vi_lq_in", "ir_lq_in", *image_keys]
    min_height = min(sample["vi_hq"].shape[-2] for sample in batch)
    min_width = min(sample["vi_hq"].shape[-1] for sample in batch)
    min_crop_height = min(crop_size, min_height)
    min_crop_width = min(crop_size, min_width)
    if max_crop_size is None:
        crop_height = min_crop_height
        crop_width = min_crop_width
    else:
        max_crop_height = min(max_crop_size, min_height)
        max_crop_width = min(max_crop_size, min_width)
        crop_height = torch.randint(
            min_crop_height, max_crop_height + 1, ()
        ).item()
        crop_width = torch.randint(
            min_crop_width, max_crop_width + 1, ()
        ).item()
    padded_height = crop_height + (-crop_height) % divisor
    padded_width = crop_width + (-crop_width) % divisor
    output = {key: [] for key in image_keys}
    output.update({"vi_lq_clip": [], "ir_lq_clip": []})

    for sample in batch:
        height, width = sample["vi_hq"].shape[-2:]
        top = torch.randint(0, height - crop_height + 1, ()).item()
        left = torch.randint(0, width - crop_width + 1, ()).item()
        cropped = {}
        for key in image_keys:
            value = sample[key][:, top : top + crop_height, left : left + crop_width]
            cropped[key] = value
            padded = F.pad(
                value,
                (0, padded_width - crop_width, 0, padded_height - crop_height),
                value=-1,
            )
            output[key].append(padded)
        clip_visible_key = "vi_lq_in" if raw else "vi_lq"
        clip_infrared_key = "ir_lq_in" if raw else "ir_lq"
        for output_key, image_key in (
            ("vi_lq_clip", clip_visible_key),
            ("ir_lq_clip", clip_infrared_key),
        ):
            output[output_key].append(
                F.interpolate(
                    cropped[image_key].unsqueeze(0),
                    size=(clip_size, clip_size),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            )
    return {key: torch.stack(values) for key, values in output.items()}
