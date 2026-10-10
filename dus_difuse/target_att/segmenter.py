"""Optional GroundingDINO + SAM text-to-mask adapter.

GroundingDINO is installed as a dependency rather than vendored into this repository.
"""

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# Prefer the bundled GroundingDINO source tree.  Keeping this explicit avoids
# depending on a globally installed package (and on a potentially incompatible
# precompiled CUDA extension from another project/environment).
_GROUNDING_DINO_ROOT = Path(__file__).resolve().parent / "GroundingDINO"
if _GROUNDING_DINO_ROOT.is_dir():
    sys.path.insert(0, str(_GROUNDING_DINO_ROOT))

from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class TargetSegmenter:
    def __init__(
        self,
        grounding_config: str,
        grounding_checkpoint: str,
        sam_checkpoint: str,
        text_encoder: str,
        device: str,
        sam_type: str = "vit_h",
    ) -> None:
        for path in (grounding_config, grounding_checkpoint, sam_checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if not Path(text_encoder).is_dir():
            raise FileNotFoundError(f"BERT text encoder directory not found: {text_encoder}")
        self.device = torch.device(device)
        args = SLConfig.fromfile(grounding_config)
        # Gradient checkpointing is unnecessary during no-grad inference and
        # newer PyTorch releases warn when it is used without gradient inputs.
        args.use_checkpoint = False
        # Override the value embedded in upstream GroundingDINO configs so text
        # inference remains offline and independent of the current directory.
        args.text_encoder_type = str(Path(text_encoder).resolve())
        self.grounding_model = build_model(args)
        checkpoint = torch.load(grounding_checkpoint, map_location="cpu", weights_only=False)
        self.grounding_model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        self.grounding_model.to(self.device).eval()
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint).to(self.device)
        self.sam = SamPredictor(sam)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _boxes(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[torch.Tensor, List[str]]:
        from PIL import Image

        caption = prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."
        image, _ = self.transform(Image.fromarray(image_rgb), None)
        outputs = self.grounding_model(image[None].to(self.device), captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0].cpu()
        boxes = outputs["pred_boxes"][0].cpu()
        keep = logits.max(dim=1).values > box_threshold
        logits, boxes = logits[keep], boxes[keep]
        tokenized = self.grounding_model.tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, self.grounding_model.tokenizer)
            for logit in logits
        ]
        return boxes, phrases

    @torch.no_grad()
    def mask(
        self,
        image: torch.Tensor,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> torch.Tensor:
        """Return a [1,H,W] binary mask for one RGB tensor in [0,1]."""
        image_rgb = (
            image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype(np.uint8)
        height, width = image_rgb.shape[:2]
        boxes, _ = self._boxes(image_rgb, prompt, box_threshold, text_threshold)
        result = torch.zeros((1, height, width), device=self.device)
        if boxes.numel() == 0:
            return result
        boxes = boxes * torch.tensor([width, height, width, height])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        self.sam.set_image(image_rgb)
        transformed = self.sam.transform.apply_boxes_torch(boxes, (height, width)).to(self.device)
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        return masks[:, 0].any(dim=0, keepdim=True).float()
