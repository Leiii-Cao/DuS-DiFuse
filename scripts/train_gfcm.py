#!/usr/bin/env python3
"""Train GFCM from SDE-sampled pseudo-supervision."""

import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch
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
from dus_difuse.losses import GFCMPseudoSupervisionLoss
from dus_difuse.model import Diffusion, DuSDiFuseModel
from dus_difuse.sampler import SpacedSampler
from dus_difuse.utils.common import instantiate_from_config, to
from scripts._common import load_state, paired_random_crop, project_path, resolve_train_paths


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
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.ckpt_every is not None:
        cfg.train.ckpt_every = args.ckpt_every
    resolve_train_paths(cfg)
    set_seed(cfg.train.seed, device_specific=True)

    checkpoint_dir = os.path.join(cfg.train.exp_dir, "checkpoints")
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(cfg.train.exp_dir)

    model: DuSDiFuseModel = instantiate_from_config(cfg.model.dus_difuse)
    model.load_pretrained_VAE_MDA(load_state(cfg.train.sd_path))
    model.load_unet_from_ckpt(load_state(cfg.train.unetresume))
    if cfg.train.skipresume:
        model.load_gfcm_from_ckpt(load_state(cfg.train.skipresume))
    for parameter in model.unet.parameters():
        parameter.requires_grad = False
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("gfcm.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Only GFCM may remain trainable; unexpected parameters: "
            + ", ".join(unexpected_trainable[:10])
        )
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    criterion = GFCMPseudoSupervisionLoss(
        lambda_luminance=cfg.train.lambda_luminance,
        lambda_chroma=cfg.train.lambda_chroma,
        lambda_gradient=cfg.train.lambda_gradient,
        lambda_ssim=cfg.train.lambda_ssim,
    )
    optimizer = torch.optim.AdamW(model.gfcm.parameters(), lr=cfg.train.learning_rate)
    dataset = FusionTrainingDataset(
        cfg.train.dataset_root,
        crop_size=cfg.train.crop_size,
        return_params=True,
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
        ),
    )
    if not len(loader):
        raise RuntimeError("No training batches; reduce batch_size or disable drop_last")

    if accelerator.is_main_process:
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        frozen_parameters = sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        )
        print(f"Dataset samples: {len(dataset):,}; batches per pass: {len(loader):,}")
        print(f"SDE checkpoint: {cfg.train.unetresume}")
        print(
            "GFCM initialization: "
            + (str(cfg.train.skipresume) if cfg.train.skipresume else "random")
        )
        print(
            f"Trainable GFCM parameters: {trainable_parameters:,}; "
            f"frozen parameters: {frozen_parameters:,}"
        )
        print(
            f"Training steps: {cfg.train.train_steps:,}; global batch size: "
            f"{cfg.train.batch_size}; sampling steps: {cfg.train.sample_steps}"
        )

    model.train().to(accelerator.device)
    diffusion.to(accelerator.device)
    criterion.to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    pure_model: DuSDiFuseModel = accelerator.unwrap_model(model)
    global_step = 0
    loss_buffer = []
    component_names = ("luminance", "chroma", "gradient", "ssim")
    component_weights = {
        "luminance": cfg.train.lambda_luminance,
        "chroma": cfg.train.lambda_chroma,
        "gradient": cfg.train.lambda_gradient,
        "ssim": cfg.train.lambda_ssim,
    }
    component_buffers = {name: [] for name in component_names}

    while global_step < cfg.train.train_steps:
        progress = tqdm(loader, disable=not accelerator.is_main_process, unit="batch")
        for batch in progress:
            batch = to(batch, accelerator.device)
            with torch.no_grad():
                condition = pure_model.prepare_condition(
                    batch["vi_lq"].float(),
                    batch["ir_lq"].float(),
                    batch["vi_lq_clip"].float(),
                    batch["ir_lq_clip"].float(),
                )
                latent_shape = (len(batch["vi_hq"]), *condition["c_img1"].shape[1:])
                visible_latent, infrared_latent = sampler.sample(
                    model=model,
                    device=accelerator.device,
                    steps=cfg.train.sample_steps,
                    x_size=latent_shape,
                    cond=condition,
                    uncond=None,
                    cfg_scale=1.0,
                    progress=False,
                )
                enhanced_visible = (pure_model.vae_decode(visible_latent) + 1) / 2
                enhanced_infrared = (pure_model.vae_decode(infrared_latent) + 1) / 2
                reduced_timestep = torch.randint(
                    cfg.train.sample_steps,
                    (latent_shape[0],),
                    device=accelerator.device,
                )
                _, _, noisy_latent, _, model_timestep = sampler.sample(
                    model=model,
                    device=accelerator.device,
                    steps=cfg.train.sample_steps,
                    x_size=latent_shape,
                    cond=condition,
                    uncond=None,
                    cfg_scale=1.0,
                    progress=False,
                    set_dual_fusion=True,
                    return_t=reduced_timestep,
                )

            prediction, _ = model(
                noisy_latent, noisy_latent, model_timestep, condition, True
            )
            clean_latent = sampler.predict_clean(noisy_latent, reduced_timestep, prediction)
            fused = (pure_model.vae_decode(clean_latent) + 1) / 2
            terms = criterion(fused, enhanced_visible, enhanced_infrared)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(terms["total"])
            gradient_norm = accelerator.clip_grad_norm_(pure_model.gfcm.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            loss_buffer.append(terms["total"].detach())
            for name in component_names:
                component_buffers[name].append(terms[name].detach())
            progress.set_description(f"step {global_step:07d} | L_con {terms['total'].item():.6f}")
            if global_step % cfg.train.log_every == 0:
                mean_loss = accelerator.gather(torch.stack(loss_buffer)).mean().item()
                loss_buffer.clear()
                mean_components = {}
                for name in component_names:
                    mean_components[name] = accelerator.gather(
                        torch.stack(component_buffers[name])
                    ).mean().item()
                    component_buffers[name].clear()
                weighted_components = {
                    name: component_weights[name] * mean_components[name]
                    for name in component_names
                }
                weighted_sum = sum(weighted_components.values())
                if writer:
                    writer.add_scalar("loss/fusion_consistency", mean_loss, global_step)
                    for name in component_names:
                        writer.add_scalar(
                            f"loss/{name}", mean_components[name], global_step
                        )
                        writer.add_scalar(
                            f"loss_weighted/{name}", weighted_components[name], global_step
                        )
                        writer.add_scalar(
                            f"loss_fraction/{name}",
                            weighted_components[name] / max(weighted_sum, 1e-12),
                            global_step,
                        )
                    breakdown = " | ".join(
                        f"{name}={weighted_components[name]:.6f} "
                        f"({100.0 * weighted_components[name] / max(weighted_sum, 1e-12):.2f}%)"
                        for name in component_names
                    )
                    print(f"step {global_step:07d} weighted loss | {breakdown}")
                    print(f"step {global_step:07d} pre-clip gradient norm: {float(gradient_norm):.6f}")
            if global_step % cfg.train.ckpt_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        pure_model.gfcm.state_dict(),
                        os.path.join(checkpoint_dir, f"{global_step:07d}.pt"),
                    )
            if global_step >= cfg.train.train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(pure_model.gfcm.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/gfcm.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--unet-resume")
    parser.add_argument("--gfcm-resume")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    main(parser.parse_args())
