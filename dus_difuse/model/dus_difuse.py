from typing import Tuple, Set, List, Dict

import torch
from torch import nn
from .controlnet import ControlledUnetModel, ControlNet
from .unet import DualEncoder_IFModel
from .gfcm import GroupwiseFusionControlModule
from .vae import AutoencoderKL
from .util import GroupNorm32
from .mda import MultimodalDegradationAwareEncoder
from .clip import FrozenOpenCLIPEmbedder_text
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook
from ..utils.common import sliding_windows, count_vram_usage, gaussian_weights

def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DuSDiFuseModel(nn.Module):
    """Diffusion Fusion, Generative Modulation, and VAE components."""

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, latent_scale_factor, use_gfcm=False,
        controlnet_cfg=None, SD_unet_cfg=None, clip_text_cfg=None
    ):
        super().__init__()
        self.unet = DualEncoder_IFModel(**unet_cfg) if unet_cfg is not None else None
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = (
            MultimodalDegradationAwareEncoder(**clip_cfg) if clip_cfg is not None else None
        )
        self.clip_text = (
            FrozenOpenCLIPEmbedder_text(**clip_text_cfg)
            if clip_text_cfg is not None
            else None
        )
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        if controlnet_cfg is not None:  
            self.controlnet = ControlNet(**controlnet_cfg)
            self.SD_unet = ControlledUnetModel(**SD_unet_cfg)
        else:
            self.controlnet = None
            self.SD_unet = None
        if use_gfcm:
            if self.unet is None:
                raise ValueError("GFCM requires unet_cfg")
            input_block_chans, time_emb_dim = self.unet.get_skip_config()
            self.gfcm = GroupwiseFusionControlModule(
                input_block_chans, time_emb_dim, enable_checkpointing=False
            )
        else:
            self.gfcm = None
            
    @torch.no_grad()
    def load_pretrained_VAE_only(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "vae": "first_stage_model",
        }
        modules = [("vae", self.vae)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key]
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_pretrained_VAE_MDA(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        """Load the SD VAE while preserving and freezing the MDA checkpoint."""
        unused, missing = self.load_pretrained_VAE_only(sd)
        self.clip.eval()
        self.clip.train = disabled_train
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        return unused, missing
        
    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.SD_unet), ("vae", self.vae), ("clip", self.clip_text)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key]
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip_text, self.SD_unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing
        
    @torch.no_grad()
    def load_unet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.unet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_gfcm_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.gfcm.load_state_dict(sd, strict=True)
        
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str], Set[str]]:
        """Initialize Generation ControlNet from the frozen SD UNet."""
        if self.controlnet is None or self.SD_unet is None:
            raise RuntimeError("Generation ControlNet and SD UNet are not configured")
        unet_sd = self.SD_unet.state_dict()
        control_sd = self.controlnet.state_dict()
        initialized_with_zeros = set()
        initialized_from_scratch = set()
        initialized = {}
        for key, control_value in control_sd.items():
            if key not in unet_sd:
                initialized[key] = control_value
                initialized_from_scratch.add(key)
                continue
            unet_value = unet_sd[key]
            if control_value.shape == unet_value.shape:
                initialized[key] = unet_value
                continue
            if control_value.ndim != 4 or control_value.shape[1] < unet_value.shape[1]:
                initialized[key] = control_value
                initialized_from_scratch.add(key)
                continue
            extra_channels = control_value.shape[1] - unet_value.shape[1]
            zeros = torch.zeros(
                (control_value.shape[0], extra_channels, *control_value.shape[2:]),
                dtype=unet_value.dtype,
                device=unet_value.device,
            )
            initialized[key] = torch.cat((unet_value, zeros), dim=1)
            initialized_with_zeros.add(key)
        self.controlnet.load_state_dict(initialized, strict=True)
        return initialized_with_zeros, initialized_from_scratch


    def vae_encode_tiled(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        Get_skip: bool = False,
    ) -> torch.Tensor:
        bs, _, h, w = image.shape
        z = torch.zeros((bs, 4, h // 8, w // 8), dtype=torch.float32, device=image.device)
        count = torch.zeros_like(z, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=image.device)
        tiles = sliding_windows(h // 8, w // 8, tile_size // 8, tile_stride // 8)
        skip_feats = []
        for hi, hi_end, wi, wi_end in tiles:
            tile_image = image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8]
            with torch.no_grad():
                encoded = self.vae_encode(tile_image, sample=sample, Get_skip=Get_skip)
                if Get_skip:
                    z_middle, skips_middle = encoded
                    skip_feats.append(skips_middle)
                else:
                    z_middle = encoded
                z[:, :, hi:hi_end, wi:wi_end] += z_middle * weights
                count[:, :, hi:hi_end, wi:wi_end] += weights
        z.div_(count)
        return z, skip_feats if Get_skip else None

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        Get_skip: bool = False,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            if Get_skip:
                z, skips = encoder(image, Get_skip)
                return z.sample() * self.scale_factor, skips
            else:
                z = encoder(image, Get_skip).sample() * self.scale_factor
                return z
        else:
            if Get_skip:
                z, skips = encoder(image, Get_skip)
                return z.mode() * self.scale_factor, skips
            else:
                z = encoder(image, Get_skip).mode() * self.scale_factor
                return z
                

    @count_vram_usage
    def vae_decode_tiled(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        skip_lq1 = None,
        skip_lq2 = None,
        drfm=None,
    ) -> torch.Tensor:
        bs, _, h, w = z.shape
        image = torch.zeros((bs, 3, h * 8, w * 8), dtype=torch.float32, device=z.device)
        count = torch.zeros_like(image, dtype=torch.float32)
        weights = gaussian_weights(tile_size * 8, tile_size * 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=z.device)
        tiles = sliding_windows(h, w, tile_size, tile_stride)
        for ind, (hi, hi_end, wi, wi_end) in enumerate(tiles):
            skip_feat1 = skip_lq1[ind] if skip_lq1 is not None else None
            skip_feat2 = skip_lq2[ind] if skip_lq2 is not None else None
            tile_z = z[:, :, hi:hi_end, wi:wi_end]
            with torch.no_grad():
                tile_z_decoded = self.vae_decode(
                    tile_z, skip_lq1=skip_feat1, skip_lq2=skip_feat2, drfm=drfm
                )
                image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += tile_z_decoded * weights
                count[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += weights
        image.div_(count)
        return image


    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
        skip_lq1 = None,
        skip_lq2 = None,
        drfm=None,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(
            z / self.scale_factor,
            skip_lq1=skip_lq1,
            skip_lq2=skip_lq2,
            drfm=drfm,
        )
        
        
        
    def prepare_condition_tiled(
        self,
        cond_img1: torch.Tensor,
        cond_img2: torch.Tensor,
        txt1: torch.Tensor,
        txt2: torch.Tensor,
        Label: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        Get_skip: bool = False,
        skip_input=None,
    ) -> Dict[str, torch.Tensor]:
        
        c_txt = self.clip_text.encode(txt) if self.clip_text is not None else None
        c_txt1=self.clip.encode(txt1)
        c_txt2=self.clip.encode(txt2)
        c_img1,skip_lq1=self.vae_encode_tiled(
            cond_img1,
            sample=False,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride = tile_stride,
            Get_skip = Get_skip,
        )
        c_label = None
        if self.clip_text is not None:
            c_label, _ = self.vae_encode_tiled(
                Label,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                Get_skip=Get_skip,
            )
        c_img2,skip_lq2=self.vae_encode_tiled(
            cond_img2,
            sample=False,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride = tile_stride,
            Get_skip = Get_skip,
        )
        return dict(
            c_txt = c_txt,
            c_label = c_label,
            c_txt1 = c_txt1,
            c_txt2 = c_txt2,
            c_img1 = c_img1,
            c_img2 = c_img2,
            skip_lq1 = skip_lq1,
            skip_lq2 = skip_lq2,
        )
    
        
        
        
        
        

    def prepare_condition(
        self,
        cond_img1: torch.Tensor,
        cond_img2: torch.Tensor,
        txt1: torch.Tensor,
        txt2: torch.Tensor,
        Label: torch.Tensor = None,
        txt: List[str] = None,
        tiled: bool = False,
        tile_size: int = -1,
        Get_skip: bool = False,
        skip_input=None,
    ) -> Dict[str, torch.Tensor]:
        if Label is None:
            Label = torch.ones_like(cond_img1)
        if txt is None:
            txt = [""] * cond_img1.shape[0]
        c_txt = self.clip_text.encode(txt) if self.clip_text is not None else None
        if Get_skip == False:
            return dict(
                c_txt=c_txt,
                c_txt1=self.clip.encode(txt1),
                c_txt2=self.clip.encode(txt2),
                c_img1=self.vae_encode(
                    cond_img1,
                    sample=False,
                    tiled=tiled,
                    tile_size=tile_size,
                ),
                c_label=(
                    self.vae_encode(
                        Label,
                        sample=False,
                        tiled=tiled,
                        tile_size=tile_size,
                    )
                    if self.clip_text is not None
                    else None
                ),
                c_img2=self.vae_encode(
                    cond_img2,
                    sample=False,
                    tiled=tiled,
                    tile_size=tile_size,
                ),
                skip_lq1=None,
                skip_lq2=None,
            )
        else:
            c_txt=c_txt
            c_txt1=self.clip.encode(txt1)
            c_txt2=self.clip.encode(txt2)
            c_img1,skip_lq1=self.vae_encode(
                cond_img1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
                Get_skip = Get_skip,
            )
            c_label = (
                self.vae_encode(
                    Label,
                    sample=False,
                    tiled=tiled,
                    tile_size=tile_size,
                )
                if self.clip_text is not None
                else None
            )
            c_img2,skip_lq2=self.vae_encode(
                cond_img2,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
                Get_skip = Get_skip,
            )
            return dict(
                c_txt = c_txt,
                c_label = c_label,
                c_txt1 = c_txt1,
                c_txt2 = c_txt2,
                c_img1 = c_img1,
                c_img2 = c_img2,
                skip_lq1 = skip_lq1,
                skip_lq2 = skip_lq2,
            )
            
        

    def prepare_generation_condition(
        self,
        control_image: torch.Tensor,
        generation_mask: torch.Tensor,
        prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        if self.clip_text is None:
            raise RuntimeError("Generation Diffusion requires clip_text_cfg")
        return {
            "c_txt": self.clip_text.encode(prompts),
            "c_img": self.vae_encode(control_image, sample=False),
            "c_label": self.vae_encode(generation_mask, sample=False),
            "_generation": True,
        }

    def forward(self, x_noisy1,x_noisy2, t, cond,  set_dual_fusion=False):
        if cond.get("_generation", False):
            return self.forward_generation(x_noisy1, t, cond)
        c_txt = cond.get("c_txt")
        c_txt1 = cond["c_txt1"]
        c_txt2 = cond["c_txt2"]
        c_img1 = cond["c_img1"]
        c_img2 = cond["c_img2"]
        x_noisy_cond1 = torch.cat((x_noisy1, c_img1), dim=1)
        x_noisy_cond2 = torch.cat((x_noisy2, c_img2), dim=1)
        eps1,eps2 = self.unet(
            x1=x_noisy_cond1,
            x2=x_noisy_cond2,
            timesteps=t,
            context1=c_txt1,
            context2=c_txt2,
            control=None,
            only_mid_control=False,
            gfcm=self.gfcm if (self.gfcm is not None and set_dual_fusion) else None
        )
        return eps1, eps2

    def forward_generation(self, x_noisy, t, cond):
        """Predict Generation Diffusion noise/velocity for one latent stream."""
        c_hint = torch.cat((cond["c_img"], cond["c_label"]), dim=1)
        control = self.controlnet(
            x=x_noisy,
            hint=c_hint,
            timesteps=t,
            context=cond["c_txt"],
        )
        control = [value * scale for value, scale in zip(control, self.control_scales)]
        return self.SD_unet(
            x=x_noisy,
            timesteps=t,
            context=cond["c_txt"],
            control=control,
            only_mid_control=False,
        )
            
    def forward_control(self, x_noisy1, x_noisy2, t, cond):
        """Apply posterior generative modulation to the shared fused stream."""
        hint = torch.cat((cond["c_img1"], cond["c_label"]), dim=1)
        control = self.controlnet(
            x=x_noisy1,
            hint=hint,
            timesteps=t,
            context=cond["c_txt"],
        )
        control = [value * scale for value, scale in zip(control, self.control_scales)]
        output = self.SD_unet(
            x=x_noisy1,
            timesteps=t,
            context=cond["c_txt"],
            control=control,
            only_mid_control=False,
        )
        return output, output

    def cast_dtype(self, dtype: torch.dtype) -> "DuSDiFuseModel":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
