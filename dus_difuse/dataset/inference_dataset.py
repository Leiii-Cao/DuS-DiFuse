from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def _pad_pair(images: List[np.ndarray], divisor: int) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    height, width = images[0].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("Visible and infrared images must have identical spatial sizes")
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        images = [
            np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            for image in images
        ]
    return images, (height, width)


class FusionInferenceDataset(Dataset):
    """Load filename-aligned visible/infrared pairs for inference."""

    def __init__(
        self,
        root: Optional[str] = None,
        visible_dir: Optional[str] = None,
        infrared_dir: Optional[str] = None,
        divisor: int = 64,
        clip_size: int = 224,
    ) -> None:
        if root:
            root_path = Path(root)
            if not visible_dir and not infrared_dir:
                directory_pairs = (("vi", "ir"), ("source_1", "source_2"))
                for visible_name, infrared_name in directory_pairs:
                    candidate_visible = root_path / visible_name
                    candidate_infrared = root_path / infrared_name
                    if candidate_visible.is_dir() and candidate_infrared.is_dir():
                        visible_dir = str(candidate_visible)
                        infrared_dir = str(candidate_infrared)
                        break

                # Keep the original paths in the error message when neither
                # supported directory layout exists.
                if not visible_dir:
                    visible_dir = str(root_path / "vi")
                    infrared_dir = str(root_path / "ir")
        if not visible_dir or not infrared_dir:
            raise ValueError("Provide --input-root or both --visible-dir and --infrared-dir")

        self.visible_dir = Path(visible_dir)
        self.infrared_dir = Path(infrared_dir)
        self.divisor = divisor
        self.clip_size = clip_size
        if not self.visible_dir.is_dir() or not self.infrared_dir.is_dir():
            raise FileNotFoundError(
                f"Expected image directories: {self.visible_dir} and {self.infrared_dir}"
            )

        visible_names = {
            path.name for path in self.visible_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        infrared_names = {
            path.name for path in self.infrared_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        self.names = sorted(visible_names & infrared_names)
        if not self.names:
            raise RuntimeError("No filename-aligned visible/infrared image pairs were found")
        missing = sorted(visible_names ^ infrared_names)
        if missing:
            print(f"Warning: ignoring {len(missing)} unpaired image(s)")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.names[index]
        vi_path = self.visible_dir / name
        ir_path = self.infrared_dir / name
        vi, ir = _read_rgb(vi_path), _read_rgb(ir_path)
        (vi, ir), original_size = _pad_pair([vi, ir], self.divisor)
        vi_tensor, ir_tensor = _to_tensor(vi), _to_tensor(ir)
        vi_clip = F.interpolate(
            vi_tensor.unsqueeze(0), size=(self.clip_size, self.clip_size),
            mode="bicubic", align_corners=False,
        ).squeeze(0)
        ir_clip = F.interpolate(
            ir_tensor.unsqueeze(0), size=(self.clip_size, self.clip_size),
            mode="bicubic", align_corners=False,
        ).squeeze(0)
        return {
            "visible": vi_tensor,
            "infrared": ir_tensor,
            "visible_clip": vi_clip,
            "infrared_clip": ir_clip,
            "name": name,
            "original_size": torch.tensor(original_size, dtype=torch.int64),
        }
