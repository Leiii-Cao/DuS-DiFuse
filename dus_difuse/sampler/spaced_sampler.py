from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.dus_difuse import DuSDiFuseModel
from ..utils.common import sliding_windows, gaussian_weights
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(Sampler):

    CONTROL_SCALE_MIN = 1.0
    CONTROL_SCALE_MAX = 1.5

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "SpacedSampler":
        super().__init__(betas, parameterization, rescale_cfg)

    @classmethod
    def adaptive_control_scale(
        cls,
        mask_coverage: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return a smooth per-image control scale based on mask coverage.

        A full-image mask uses 1.0. As the mask gets smaller, smoothstep
        increases the scale toward 1.5 without discontinuities at either end.
        """
        batch_size = reference.shape[0]
        if mask_coverage is None:
            coverage = torch.ones(
                batch_size, device=reference.device, dtype=reference.dtype
            )
        else:
            coverage = torch.as_tensor(
                mask_coverage, device=reference.device, dtype=reference.dtype
            ).flatten()
            if coverage.numel() == 1 and batch_size > 1:
                coverage = coverage.expand(batch_size)
            elif coverage.numel() != batch_size:
                raise ValueError(
                    "generation mask coverage must contain one value per image"
                )

        inverse_coverage = 1.0 - coverage.clamp(0.0, 1.0)
        smooth_weight = inverse_coverage.square() * (3.0 - 2.0 * inverse_coverage)
        scale = cls.CONTROL_SCALE_MIN + (
            cls.CONTROL_SCALE_MAX - cls.CONTROL_SCALE_MIN
        ) * smooth_weight
        return scale.view(batch_size, 1, 1, 1)

    def make_schedule(self, num_steps: int) -> None:
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(
            sorted(list(used_timesteps)), dtype=np.int32
        )  # e.g. [0, 10, 20, ...]
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def predict_clean(
        self, noisy: torch.Tensor, timestep: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        if self.parameterization == "eps":
            return self._predict_xstart_from_eps(noisy, timestep, model_output)
        return self._predict_xstart_from_v(noisy, timestep, model_output)

    def apply_model_tiled(
        self,
        model: DuSDiFuseModel,
        x1: torch.Tensor,
        x2: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        set_dual_fusion: bool,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        fix_init_noise: bool
    ) -> torch.Tensor:
        _, _, h, w = x1.shape
        tiles_latent = tqdm(sliding_windows(h, w, tile_size // 8, tile_stride // 8), unit="tile", leave=False)
        tiles_img = tqdm(sliding_windows(h * 8, w * 8, tile_size, tile_stride), unit="tile", leave=False)
        eps1 = torch.zeros_like(x1)
        eps2 = torch.zeros_like(x2)
        count1 = torch.zeros_like(x1, dtype=torch.float32)
        count2 = torch.zeros_like(x2, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=x1.device)
        for (hi_latent, hi_end_latent, wi_latent, wi_end_latent), (hi_img, hi_end_img, wi_img, wi_end_img) in zip(tiles_latent, tiles_img):
            tiles_latent.set_description(f"Process tile ({hi_latent} {hi_end_latent}), ({wi_latent} {wi_end_latent})")
            if fix_init_noise:
                tile_x1 = x1[:, :, 0:64, 0:64]
                tile_x2 = x2[:, :, 0:64, 0:64]
            else:
                tile_x1 = x1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent]
                tile_x2 = x2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent]
            tile_cond = {
                "c_txt1":cond["c_txt1"],
                "c_txt2":cond["c_txt2"],
                "c_txt":cond["c_txt"],
                "c_label": (
                    cond["c_label"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent]
                    if cond.get("c_label") is not None
                    else None
                ),
                "c_img1":cond["c_img1"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent],
                "c_img2":cond["c_img2"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent],
            }
            tile_uncond = None
            tile_eps1, tile_eps2 = self.apply_model(model,tile_x1, tile_x2, model_t, tile_cond, tile_uncond, cfg_scale,set_dual_fusion)
            eps1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += tile_eps1 * weights
            eps2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += tile_eps2 * weights
            count1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += weights
            count2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += weights
        eps1.div_(count1)
        eps2.div_(count2)
        return eps1,eps2
        


    def apply_model(
        self,
        model: DuSDiFuseModel,
        x1: torch.Tensor,
        x2: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        set_dual_fusion: bool,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output1, model_output2 = model(x1, x2, model_t, cond, set_dual_fusion)
        else:
            model_cond1, model_cond2 = model(x1, x2, model_t, cond, set_dual_fusion)
            model_uncond1, model_uncond2 = model(x1, x2, model_t, uncond, set_dual_fusion)
            model_output1 = model_uncond1 + cfg_scale * (model_cond1 - model_uncond1)
            model_output2 = model_uncond2 + cfg_scale * (model_cond2 - model_uncond2)
        return model_output1, model_output2

    def apply_control_tiled(
        self,
        model: DuSDiFuseModel,
        x1: torch.Tensor,
        x2: torch.Tensor,
        model_t: torch.Tensor,
        control_cond: Dict[str, torch.Tensor],
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        fix_init_noise: bool
    ) -> torch.Tensor:
        _, _, h, w = x1.shape
        tiles_latent = tqdm(sliding_windows(h, w, tile_size // 8, tile_stride // 8), unit="tile", leave=False)
        tiles_img = tqdm(sliding_windows(h * 8, w * 8, tile_size, tile_stride), unit="tile", leave=False)
        eps1 = torch.zeros_like(x1)
        eps2 = torch.zeros_like(x2)
        count1 = torch.zeros_like(x1, dtype=torch.float32)
        count2 = torch.zeros_like(x2, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=x1.device)
        for (hi_latent, hi_end_latent, wi_latent, wi_end_latent), (hi_img, hi_end_img, wi_img, wi_end_img) in zip(tiles_latent, tiles_img):
            tiles_latent.set_description(f"Process tile ({hi_latent} {hi_end_latent}), ({wi_latent} {wi_end_latent})")
            if fix_init_noise:
                tile_x1 = x1[:, :, 0:64, 0:64]
                tile_x2 = x2[:, :, 0:64, 0:64]
            else:
                tile_x1 = x1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent]
                tile_x2 = x2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent]
            tile_cond = {
                "c_txt1":control_cond["c_txt1"],
                "c_txt2":control_cond["c_txt2"],
                "c_txt":control_cond["c_txt"],
                "c_label":control_cond["c_label"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent],
                "c_img1":control_cond["c_img1"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent],
                "c_img2":control_cond["c_img2"][:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent],
            }
            
            tile_eps1, tile_eps2 = model.forward_control(tile_x1, tile_x2, model_t, tile_cond)
            eps1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += tile_eps1 * weights
            eps2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += tile_eps2 * weights
            count1[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += weights
            count2[:, :, hi_latent:hi_end_latent, wi_latent:wi_end_latent] += weights
        eps1.div_(count1)
        eps2.div_(count2)
        return eps1,eps2

    @torch.no_grad()
    def p_sample(
        self,
        model: DuSDiFuseModel,
        x1: torch.Tensor,
        x2: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        set_dual_fusion: bool,
        use_control: bool,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        control_scale: float | torch.Tensor = 1.0,
        fix_init_noise: bool=False,
    ) -> torch.Tensor:
        # predict x_0
        if tiled:
            model_output1, model_output2 = self.apply_model_tiled(model, x1, x2, model_t, cond, uncond, cfg_scale, set_dual_fusion,tiled,tile_size,tile_stride,fix_init_noise)
        else:
            model_output1, model_output2 = self.apply_model(model, x1, x2, model_t, cond, uncond, cfg_scale, set_dual_fusion)
        if self.parameterization == "eps":
            pred_x01 = self._predict_xstart_from_eps(x1, t, model_output1)
            pred_x02 = self._predict_xstart_from_eps(x2, t, model_output2)
            if use_control:
                cond_control = cond.copy()
                cond_control["c_img1"] = pred_x01
                cond_control["c_img2"] = pred_x02
                eps1,eps2 = model.forward_control(x1, x2, model_t, cond_control)
                pred_x01 = self._predict_xstart_from_eps(x1, t, eps1)
                pred_x02 = self._predict_xstart_from_eps(x2, t, eps2)
            else:
                pass
        else:
            pred_x01_pre = self._predict_xstart_from_v(x1, t, model_output1)
            pred_x02_pre = self._predict_xstart_from_v(x2, t, model_output2)
            if use_control:
                cond_control = cond.copy()
                cond_control["c_img1"] = pred_x01_pre
                cond_control["c_img2"] = pred_x02_pre
                if tiled:
                    eps1,eps2 = self.apply_control_tiled(model,x1, x2, model_t, cond_control,tiled,tile_size,tile_stride,fix_init_noise)
                else:
                    eps1,eps2 = model.forward_control(x1, x2, model_t, cond_control)
                pred_x01 = self._predict_xstart_from_v(x1, t, eps1)
                pred_x02 = self._predict_xstart_from_v(x2, t, eps2)
                pred_x01 = (pred_x01 - pred_x01_pre) * control_scale + pred_x01_pre
                pred_x02 = (pred_x02 - pred_x02_pre) * control_scale + pred_x02_pre
            else:
                pred_x01 = pred_x01_pre
                pred_x02 = pred_x02_pre
        # calculate mean and variance of next state
        mean1, variance1 = self.q_posterior_mean_variance(pred_x01, x1, t)
        mean2, variance2 = self.q_posterior_mean_variance(pred_x02, x2, t)
        # sample next state
        noise = torch.randn_like(x1)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x1.shape) - 1)))
        x_prev1 = mean1 + nonzero_mask * torch.sqrt(variance1) * noise
        x_prev2 = mean2 + nonzero_mask * torch.sqrt(variance2) * noise


        return x_prev1, x_prev2

    @torch.no_grad()
    def sample(
        self,
        model: DuSDiFuseModel,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        set_dual_fusion = False,
        return_t = None,
        use_control = False,
        generation_mask_coverage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x1 = x_T
        x2 = x_T
        control_scale = self.adaptive_control_scale(
            generation_mask_coverage if use_control else None, x1
        )
        if return_t is not None:
            x1_return = torch.zeros_like(x_T)
            x2_return = torch.zeros_like(x_T)
            model_t_return = torch.full((x_size[0],), 0, device=device, dtype=torch.long)
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]
        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            if return_t is not None and (return_t == t).any():
                mask = (t == return_t)
                x1_return[mask] = x1[mask]
                x2_return[mask] = x2[mask]
                model_t_return[mask] = model_t[mask]
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x1, x2 = self.p_sample(
                model,
                x1,
                x2,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
                set_dual_fusion,
                use_control,
                tiled,
                tile_size,
                tile_stride,
                control_scale,
                fix_init_noise=True if i == 0 else False
            )
        if return_t is not None:
            return x1, x2, x1_return, x2_return, model_t_return
        else:
            return x1, x2
