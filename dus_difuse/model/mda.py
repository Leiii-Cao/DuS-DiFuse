import torch
import torch.nn as nn
from .open_clip import CLIP

class MultimodalDegradationAwareEncoder(nn.Module):
    """CLIP visual encoder adapted to degradation type and severity."""
    def __init__(
        self, embed_dim, vision_cfg, text_cfg, pretrained_path=None, num_classes=13
    ):
        super().__init__()
        model = CLIP(embed_dim, dict(vision_cfg), dict(text_cfg))
        for attr in [
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
        ]:
            if hasattr(model, attr):
                delattr(model, attr)
        self.model = model
    
        visual_dim = self.model.visual.output_dim

        self.classifier_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, num_classes),
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, 1),
        )

        if pretrained_path is not None:
            self.load_pretrained(filtered_path=pretrained_path)

    def load_pretrained(self, filtered_path):
        """Load either a training checkpoint or a plain OpenCLIP state dict."""
        print(f"Loading pretrained checkpoint: {filtered_path}")
        ckpt = torch.load(
            filtered_path, map_location="cpu", weights_only=False, mmap=True
        )
    
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    
        model_dict = self.state_dict()
    
        load_dict = {}
        unexpected_keys = []
        shape_mismatch = []
    
        for source_key, value in state_dict.items():
            source_key = source_key.removeprefix("module.")
            candidates = [source_key]
            if not source_key.startswith("model."):
                candidates.append(f"model.{source_key}")
            target_key = next((key for key in candidates if key in model_dict), None)
            if target_key is not None:
                if model_dict[target_key].shape == value.shape:
                    load_dict[target_key] = value
                else:
                    shape_mismatch.append(source_key)
            else:
                unexpected_keys.append(source_key)
    
        missing_keys = [k for k in model_dict.keys() if k not in load_dict]

        print(f"Loaded keys: {len(load_dict)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"Shape mismatch keys: {len(shape_mismatch)}")
    
        if len(unexpected_keys) > 0:
            print("Example unexpected:", unexpected_keys[:5])
        if len(shape_mismatch) > 0:
            print("Example mismatch:", shape_mismatch[:5])
    
        visual_keys = [key for key in load_dict if key.startswith("model.visual.")]
        if not visual_keys:
            raise RuntimeError(
                f"Checkpoint {filtered_path} did not contain compatible visual encoder weights"
            )

        model_dict.update(load_dict)
        self.load_state_dict(model_dict, strict=False)



    def forward(self, img):
        feat = self.model.encode_image(img)
        cls = self.classifier_head(feat)
        reg = self.regression_head(feat)
        return cls, reg

    def encode(self, image, pooling=False) -> torch.Tensor:
        return self.model.encode_image(image, Pooling=pooling)

