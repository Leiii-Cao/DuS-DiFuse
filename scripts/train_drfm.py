#!/usr/bin/env python3
"""Train the Detail-Restoration Fidelity Module (Eqs. 26-28)."""

import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch
from torch import nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dus_difuse.dataset import FusionTrainingDataset
from dus_difuse.losses import DetailRestorationFidelityLoss
from dus_difuse.model import DetailRestorationFidelityModule, Diffusion, DuSDiFuseModel
from dus_difuse.sampler import SpacedSampler
from dus_difuse.utils.common import instantiate_from_config, to
from scripts._common import load_state, paired_random_crop, project_path, resolve_train_paths


class DRFMDecoder(nn.Module):
    """Expose the complete DRFM-assisted decode as one DDP forward call."""

    def __init__(self, drfm: DetailRestorationFidelityModule) -> None:
        super().__init__()
        self.drfm = drfm

    def forward(self, decode, latent, visible_skips, infrared_skips):
        return decode(
            latent,
            skip_lq1=visible_skips,
            skip_lq2=infrared_skips,
            drfm=self.drfm,
        )


def main(args) -> None:
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True)
    )
    cfg = OmegaConf.load(project_path(args.config))
    if args.data_root:
        cfg.train.dataset_root = args.data_root
    if args.output_dir:
        cfg.train.exp_dir = args.output_dir
    if args.unet_resume:
        cfg.train.unetresume = args.unet_resume
    if args.gfcm_resume:
        cfg.train.skipresume = args.gfcm_resume
    if args.drfm_resume:
        cfg.train.resume = args.drfm_resume
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.ckpt_every is not None:
        cfg.train.ckpt_every = args.ckpt_every
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    resolve_train_paths(cfg)
    set_seed(cfg.train.seed, device_specific=True)

    if cfg.train.batch_size % accelerator.num_processes:
        raise ValueError(
            f"Global batch size {cfg.train.batch_size} must be divisible by "
            f"the process count {accelerator.num_processes} when split_batches=True"
        )

    checkpoint_dir = os.path.join(cfg.train.exp_dir, "checkpoints")
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(cfg.train.exp_dir)

    model: DuSDiFuseModel = instantiate_from_config(cfg.model.dus_difuse)
    _, missing_vae = model.load_pretrained_VAE_only(load_state(cfg.train.sd_path))
    if missing_vae:
        raise RuntimeError(
            f"Stable Diffusion checkpoint is missing {len(missing_vae)} VAE keys"
        )
    model.load_unet_from_ckpt(load_state(cfg.train.unetresume))
    model.load_gfcm_from_ckpt(load_state(cfg.train.skipresume))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(accelerator.device)
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(accelerator.device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    drfm = DetailRestorationFidelityModule(cfg.model.drfm.params)
    if cfg.train.get("resume"):
        drfm.load_state_dict(load_state(cfg.train.resume), strict=True)
    drfm_decoder = DRFMDecoder(drfm)
    criterion = DetailRestorationFidelityLoss(
        lambda_luminance=cfg.train.lambda_luminance,
        lambda_chroma=cfg.train.lambda_chroma,
        lambda_gradient=cfg.train.lambda_gradient,
    ).to(accelerator.device)
    optimizer = torch.optim.AdamW(drfm_decoder.parameters(), lr=cfg.train.learning_rate)
    dataset = FusionTrainingDataset(
        cfg.train.dataset_root,
        crop_size=cfg.train.crop_size,
        return_params=True,
        om_degradation=True,
        motion_kernel_path=cfg.train.motion_kernel_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=cfg.train.drop_last,
        pin_memory=True,
        collate_fn=partial(
            paired_random_crop,
            crop_size=cfg.train.crop_size,
            divisor=cfg.train.divisor,
            clip_size=cfg.train.clip_size,
            raw=True,
            max_crop_size=cfg.train.max_crop_size,
        ),
    )
    if not len(loader):
        raise RuntimeError("No training batches; reduce batch_size or disable drop_last")

    if accelerator.is_main_process:
        trainable_parameters = sum(
            parameter.numel() for parameter in drfm.parameters()
        )
        frozen_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"Dataset samples: {len(dataset):,}; batches per pass: {len(loader):,}")
        print(f"SDE checkpoint: {cfg.train.unetresume}")
        print(f"GFCM checkpoint: {cfg.train.skipresume}")
        print(
            "DRFM initialization: "
            + (str(cfg.train.resume) if cfg.train.get("resume") else "random")
        )
        print(
            f"Trainable DRFM parameters: {trainable_parameters:,}; "
            f"frozen backbone parameters: {frozen_parameters:,}"
        )
        print(
            f"Training steps: {cfg.train.train_steps:,}; global batch size: "
            f"{cfg.train.batch_size}; per-process batch size: "
            f"{cfg.train.batch_size // accelerator.num_processes}; "
            f"sampling steps: {cfg.train.sample_steps}"
        )

    drfm_decoder.train().to(accelerator.device)
    drfm_decoder, optimizer, loader = accelerator.prepare(
        drfm_decoder, optimizer, loader
    )
    pure_drfm_decoder: DRFMDecoder = accelerator.unwrap_model(drfm_decoder)
    global_step = 0
    component_names = ("total", "luminance", "chroma", "gradient")
    loss_buffers = {name: [] for name in component_names}

    while global_step < cfg.train.train_steps:
        progress = tqdm(loader, disable=not accelerator.is_main_process, unit="batch")
        for batch in progress:
            batch = to(batch, accelerator.device)
            with torch.no_grad():
                condition = model.prepare_condition(
                    batch["vi_lq_in"].float(),
                    batch["ir_lq_in"].float(),
                    batch["vi_lq_clip"].float(),
                    batch["ir_lq_clip"].float(),
                )
                latent_shape = (len(batch["vi_hq"]), *condition["c_img1"].shape[1:])
                visible_fused_latent, infrared_fused_latent = sampler.sample(
                    model=model,
                    device=accelerator.device,
                    steps=cfg.train.sample_steps,
                    x_size=latent_shape,
                    cond=condition,
                    uncond=None,
                    cfg_scale=1.0,
                    progress=False,
                    set_dual_fusion=True,
                )
                if not torch.equal(visible_fused_latent, infrared_fused_latent):
                    raise RuntimeError(
                        "Dual-fusion sampling returned different latent tensors"
                    )
                _, visible_skips = model.vae_encode(batch["vi_lq"].float(), Get_skip=True)
                _, infrared_skips = model.vae_encode(batch["ir_lq"].float(), Get_skip=True)
            reconstruction = (
                drfm_decoder(
                    model.vae_decode,
                    visible_fused_latent,
                    visible_skips,
                    infrared_skips,
                )
                + 1
            ) / 2
            terms = criterion(
                reconstruction,
                (batch["vi_hq"].float() + 1) / 2,
                (batch["ir_hq"].float() + 1) / 2,
            )
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(terms["total"])
            gradient_norm = accelerator.clip_grad_norm_(
                drfm_decoder.parameters(), 1.0
            )
            optimizer.step()

            global_step += 1
            for name in component_names:
                loss_buffers[name].append(terms[name].detach())
            progress.set_description(f"step {global_step:07d} | L_DRFM {terms['total'].item():.6f}")
            if global_step % cfg.train.log_every == 0:
                means = {
                    name: accelerator.gather(torch.stack(values)).mean().item()
                    for name, values in loss_buffers.items()
                }
                for values in loss_buffers.values():
                    values.clear()
                if writer:
                    writer.add_scalar("loss/drfm", means["total"], global_step)
                    writer.add_scalar("loss/luminance", means["luminance"], global_step)
                    writer.add_scalar("loss/chroma", means["chroma"], global_step)
                    writer.add_scalar("loss/gradient", means["gradient"], global_step)
                    writer.add_scalar(
                        "loss_weighted/luminance",
                        cfg.train.lambda_luminance * means["luminance"],
                        global_step,
                    )
                    writer.add_scalar(
                        "loss_weighted/chroma",
                        cfg.train.lambda_chroma * means["chroma"],
                        global_step,
                    )
                    writer.add_scalar(
                        "loss_weighted/gradient",
                        cfg.train.lambda_gradient * means["gradient"],
                        global_step,
                    )
                    writer.add_scalar(
                        "optimization/pre_clip_gradient_norm",
                        float(gradient_norm),
                        global_step,
                    )
                    print(
                        f"step {global_step:07d} global mean: {means['total']:.6f} | "
                        f"Y: {cfg.train.lambda_luminance * means['luminance']:.6f} | "
                        f"CbCr: {cfg.train.lambda_chroma * means['chroma']:.6f} | "
                        f"grad: {cfg.train.lambda_gradient * means['gradient']:.6f}"
                    )
            if global_step % cfg.train.ckpt_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        pure_drfm_decoder.drfm.state_dict(),
                        os.path.join(checkpoint_dir, f"{global_step:07d}.pt"),
                    )
            if global_step >= cfg.train.train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            pure_drfm_decoder.drfm.state_dict(),
            os.path.join(checkpoint_dir, "final.pt"),
        )
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/drfm.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--unet-resume")
    parser.add_argument("--gfcm-resume")
    parser.add_argument("--drfm-resume")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    main(parser.parse_args())
