"""Training data for the optional Generation Diffusion stage."""

from pathlib import Path
import math
import random
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import (
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_mixed_kernels,
)
from .motionblur.motionblur import Kernel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    95: "building",
    105: "clouds",
    141: "plant",
    148: "road",
    156: "sky",
    168: "tree",
}


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) * 2.0 - 1.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


class GenerationDataset(Dataset):
    """Create paired restoration targets and text/mask conditions.

    The control image is degraded globally. The target is clean inside the
    sampled semantic mask and remains degraded outside it. An all-one mask
    therefore trains global restoration, while a semantic mask trains local
    text-guided restoration.
    """

    def __init__(
        self,
        image_roots: Sequence[str],
        image_ratios: Sequence[int],
        label_root: Optional[str] = None,
        class_names: Optional[Mapping] = None,
        label_all_one_prob: float = 0.5,
        prompt_drop_prob: float = 0.2,
        mask_drop_prob: float = 0.9,
        blur_kernel_size: int = 41,
        kernel_list: Sequence[str] = ("iso", "aniso"),
        kernel_prob: Sequence[float] = (0.5, 0.5),
        blur_sigma: Sequence[float] = (0.1, 8.0),
        downsample_range: Sequence[float] = (0.8, 8.0),
        noise_range: Sequence[float] = (0.0, 8.0),
        jpeg_range: Sequence[float] = (60.0, 100.0),
        motion_kernel_path: Optional[str] = None,
        motion_prob: Sequence[float] = (0.25, 0.005),
        motion_kernel_range: Sequence[int] = (1, 9),
        motion_intensity_range: Sequence[float] = (0.0, 1.0),
    ) -> None:
        if len(image_roots) != len(image_ratios):
            raise ValueError("image_roots and image_ratios must have equal length")
        self.samples = []
        for root_value, repeat in zip(image_roots, image_ratios):
            root = Path(root_value)
            if not root.is_dir():
                raise FileNotFoundError(f"Generation image directory not found: {root}")
            files = sorted(
                path for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend(files * int(repeat))
        if not self.samples:
            raise RuntimeError("Generation dataset contains no supported images")

        self.label_root = Path(label_root) if label_root else None
        raw_names = class_names or DEFAULT_CLASS_NAMES
        self.class_names: Dict[int, str] = {
            int(key): str(value) for key, value in raw_names.items()
        }
        self.label_all_one_prob = float(label_all_one_prob)
        self.prompt_drop_prob = float(prompt_drop_prob)
        self.mask_drop_prob = float(mask_drop_prob)
        self.blur_kernel_size = int(blur_kernel_size)
        self.kernel_list = list(kernel_list)
        self.kernel_prob = list(kernel_prob)
        self.blur_sigma = list(blur_sigma)
        self.downsample_range = list(downsample_range)
        self.noise_range = list(noise_range)
        self.jpeg_range = list(jpeg_range)
        self.motion_prob = list(motion_prob)
        self.motion_kernel_range = list(motion_kernel_range)
        self.motion_intensity_range = list(motion_intensity_range)
        self.motion_kernels = (
            torch.load(motion_kernel_path, map_location="cpu", weights_only=False)
            if motion_kernel_path
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_mask_and_prompt(self, image_path: Path, height: int, width: int):
        mask = np.zeros((height, width), dtype=np.float32)
        prompt = ""
        label_path = self.label_root / f"{image_path.stem}.png" if self.label_root else None
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED) if label_path else None
        if label is not None:
            if label.ndim == 3:
                label = label[..., 0]
            if label.shape != (height, width):
                label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
            candidates = [
                int(value) for value in np.unique(label)
                if int(value) != 255 and int(value) in self.class_names
            ]
            if candidates:
                class_id = random.choice(candidates)
                mask[label == class_id] = 1.0
                prompt = self.class_names[class_id]
            elif random.random() < self.label_all_one_prob:
                mask.fill(1.0)
        elif random.random() < self.label_all_one_prob:
            mask.fill(1.0)
        return mask, prompt

    def _motion_blur(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.motion_prob[0]:
            return image
        if self.motion_kernels is not None and random.random() < self.motion_prob[1]:
            key = f"{random.randint(0, 31):02d}"
            kernel = np.asarray(self.motion_kernels[key])
        else:
            size = random.randint(*self.motion_kernel_range)
            intensity = random.uniform(*self.motion_intensity_range)
            kernel = Kernel(size=(size, size), intensity=intensity).kernelMatrix
        return cv2.filter2D(image, -1, kernel)

    def _degrade(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        heavy = random.random() >= 0.3
        factor = 1.0 if heavy else 0.5
        sigma = [self.blur_sigma[0], self.blur_sigma[1] * factor]
        downsample = [self.downsample_range[0], self.downsample_range[1] * factor]
        noise = [self.noise_range[0], self.noise_range[1] * factor]
        jpeg = self.jpeg_range if heavy else [min(self.jpeg_range[0] * 1.2, 100), self.jpeg_range[1]]

        degraded = self._motion_blur(image)
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            sigma,
            sigma,
            [-math.pi, math.pi],
            noise_range=None,
        )
        degraded = cv2.filter2D(degraded, -1, kernel)
        scale = random.uniform(*downsample)
        small_w, small_h = max(1, int(width / scale)), max(1, int(height / scale))
        degraded = cv2.resize(degraded, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        degraded = random_add_gaussian_noise(degraded, noise)
        degraded = random_add_jpg_compression(degraded, jpeg)
        degraded = cv2.resize(degraded, (width, height), interpolation=cv2.INTER_LINEAR)

        if random.random() < 0.8:
            if random.random() < 0.7:
                degraded = degraded * random.uniform(0.7, 1.0) + random.uniform(-0.03, 0.03)
            if random.random() < 0.6:
                hsv = cv2.cvtColor(np.uint8(np.clip(degraded, 0, 1) * 255), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= random.uniform(0.6, 1.0)
                hsv[..., 1] += np.random.randn(height, width) * random.uniform(0.0, 0.02) * 255
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                degraded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
            if random.random() < 0.5:
                std = random.uniform(0.0, 0.05)
                degraded += np.random.randn(*degraded.shape) * std * np.sqrt(np.maximum(degraded, 1e-6))
        return np.clip(degraded, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        height, width = image.shape[:2]
        mask, prompt = self._read_mask_and_prompt(image_path, height, width)
        degraded = self._degrade(image.copy())

        blend_mask = cv2.GaussianBlur(mask, (15, 15), sigmaX=0)[..., None]
        target = image * blend_mask + degraded * (1.0 - blend_mask)

        kernel_size = random.randint(3, 10)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = (mask > 0.5).astype(np.uint8)
        binary = cv2.dilate(binary, kernel) if random.random() < 0.5 else cv2.erode(binary, kernel)
        if prompt and binary.any():
            if random.random() < self.prompt_drop_prob:
                prompt = ""
            elif random.random() < self.mask_drop_prob:
                binary.fill(0)
        mask_rgb = np.repeat(binary[..., None], 3, axis=2).astype(np.float32)
        return {
            "target": _to_tensor(target),
            "control": _to_tensor(degraded),
            "mask": _to_tensor(mask_rgb),
            "prompt": prompt,
            "name": image_path.name,
        }
