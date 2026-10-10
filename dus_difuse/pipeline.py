from typing import Optional, Tuple

import torch

from .model import Diffusion, DuSDiFuseModel
from .sampler import SpacedSampler


class DuSDiFusePipeline:
    """DuS-DiFuse inference with optional posterior generative modulation."""

    def __init__(self, model: DuSDiFuseModel, diffusion: Diffusion, device: str) -> None:
        self.model = model
        self.diffusion = diffusion
        self.device = device

    @torch.no_grad()
    def run(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        steps: int,
        visible_clip: torch.Tensor,
        infrared_clip: torch.Tensor,
        generation_mask: torch.Tensor,
        prompt: str = "",
        cfg_scale: float = 1.0,
        start_point: str = "random",
        fusion: bool = True,
        generation: bool = False,
        drfm=None,
        return_latent: bool = False,
        tiled: bool = False,
        tile_size: int = 512,
        tile_stride: int = 384,
        generation_mask_coverage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = visible.shape[0]
        get_skips = drfm is not None
        prompt_batch = [prompt] * batch_size

        if tiled:
            condition = self.model.prepare_condition_tiled(
                visible,
                infrared,
                visible_clip,
                infrared_clip,
                generation_mask.to(visible.device),
                prompt_batch,
                tiled=True,
                tile_size=tile_size,
                tile_stride=tile_stride,
                Get_skip=get_skips,
            )
        else:
            condition = self.model.prepare_condition(
                visible,
                infrared,
                visible_clip,
                infrared_clip,
                generation_mask.to(visible.device),
                prompt_batch,
                Get_skip=get_skips,
            )

        _, _, latent_h, latent_w = condition["c_img1"].shape
        if start_point == "cond":
            clean_start = torch.maximum(condition["c_img1"], condition["c_img2"])
            latent = self.diffusion.q_sample(
                clean_start,
                torch.full(
                    (batch_size,),
                    self.diffusion.num_timesteps - 1,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.randn_like(clean_start),
            )
        elif start_point == "random":
            latent = torch.randn(
                (batch_size, 4, latent_h, latent_w),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown start point: {start_point}")

        sampler = SpacedSampler(
            self.diffusion.betas,
            self.diffusion.parameterization,
            rescale_cfg=False,
        )
        visible_latent, infrared_latent = sampler.sample(
            model=self.model,
            device=self.device,
            steps=steps,
            x_size=(batch_size, 4, latent_h, latent_w),
            cond=condition,
            uncond=None,
            cfg_scale=cfg_scale,
            x_T=latent,
            progress=True,
            set_dual_fusion=fusion,
            use_control=generation,
            generation_mask_coverage=generation_mask_coverage,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if return_latent:
            return visible_latent, infrared_latent

        if tiled:
            output = self.model.vae_decode_tiled(
                visible_latent,
                skip_lq1=condition["skip_lq1"],
                skip_lq2=condition["skip_lq2"],
                drfm=drfm,
                tiled=True,
                tile_size=tile_size // 8,
                tile_stride=tile_stride // 8,
            )
        else:
            output = self.model.vae_decode(
                visible_latent,
                skip_lq1=condition["skip_lq1"],
                skip_lq2=condition["skip_lq2"],
                drfm=drfm,
            )
        return output, output
