"""Batch collation for Generative Modulation training."""

import torch
import torch.nn.functional as F


def collate_generation(batch, divisor=64, max_size=640):
    resized = []
    for sample in batch:
        height, width = sample["target"].shape[-2:]
        scale = min(1.0, max_size / max(height, width))
        output_size = (max(1, int(height * scale)), max(1, int(width * scale)))
        item = dict(sample)
        if output_size != (height, width):
            for key in ("target", "control"):
                item[key] = F.interpolate(
                    sample[key].unsqueeze(0),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            item["mask"] = F.interpolate(
                sample["mask"].unsqueeze(0), size=output_size, mode="nearest"
            ).squeeze(0)
        resized.append(item)

    target_height = max(item["target"].shape[-2] for item in resized)
    target_width = max(item["target"].shape[-1] for item in resized)
    target_height += (-target_height) % divisor
    target_width += (-target_width) % divisor

    def pad(tensor):
        return F.pad(
            tensor,
            (0, target_width - tensor.shape[-1], 0, target_height - tensor.shape[-2]),
            value=-1,
        )

    return {
        **{
            key: torch.stack([pad(item[key]) for item in resized])
            for key in ("target", "control", "mask")
        },
        "prompt": [item["prompt"] for item in resized],
        "name": [item["name"] for item in resized],
    }
