#!/usr/bin/env python3
"""Train the dual-stream degradation-decoupling diffusion model (Eq. 8)."""

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
from dus_difuse.model import Diffusion, DuSDiFuseModel
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
    if cfg.train.unetresume:
        model.load_unet_from_ckpt(load_state(cfg.train.unetresume))
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("unet.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Only the degradation-decoupling UNet may remain trainable; "
            "unexpected parameters: " + ", ".join(unexpected_trainable[:10])
        )
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=cfg.train.learning_rate)
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
        print(
            "UNet initialization: "
            + (str(cfg.train.unetresume) if cfg.train.unetresume else "random")
        )
        print(
            f"Trainable UNet parameters: {trainable_parameters:,}; "
            f"frozen parameters: {frozen_parameters:,}"
        )
        print(
            f"Training steps: {cfg.train.train_steps:,}; global batch size: "
            f"{cfg.train.batch_size}"
        )

    model.train().to(accelerator.device)
    diffusion.to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    pure_model: DuSDiFuseModel = accelerator.unwrap_model(model)
    global_step = 0
    losses = []

    while global_step < cfg.train.train_steps:
        progress = tqdm(loader, disable=not accelerator.is_main_process, unit="batch")
        for batch in progress:
            batch = to(batch, accelerator.device)
            visible_target = batch["vi_hq"].float().contiguous()
            infrared_target = batch["ir_hq"].float().contiguous()
            with torch.no_grad():
                visible_latent = pure_model.vae_encode(visible_target)
                infrared_latent = pure_model.vae_encode(infrared_target)
                condition = pure_model.prepare_condition(
                    batch["vi_lq"].float(),
                    batch["ir_lq"].float(),
                    batch["vi_lq_clip"].float(),
                    batch["ir_lq_clip"].float(),
                )
            timestep = torch.randint(
                diffusion.num_timesteps,
                (visible_latent.shape[0],),
                device=accelerator.device,
            )
            loss = diffusion.diffusion_fusion_loss(
                model, visible_latent, infrared_latent, timestep, condition
            )
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            gradient_norm = accelerator.clip_grad_norm_(pure_model.unet.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            losses.append(loss.detach())
            progress.set_description(f"step {global_step:07d} | L_diff {loss.item():.6f}")
            if global_step % cfg.train.log_every == 0:
                mean_loss = accelerator.gather(torch.stack(losses)).mean().item()
                losses.clear()
                if writer:
                    writer.add_scalar("loss/diffusion_fusion", mean_loss, global_step)
                    writer.add_scalar(
                        "optimization/pre_clip_gradient_norm",
                        float(gradient_norm),
                        global_step,
                    )
                    print(
                        f"step {global_step:07d} global mean loss: {mean_loss:.6f} | "
                        f"pre-clip gradient norm: {float(gradient_norm):.6f}"
                    )
            if global_step % cfg.train.ckpt_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        pure_model.unet.state_dict(),
                        os.path.join(checkpoint_dir, f"{global_step:07d}.pt"),
                    )
            if global_step >= cfg.train.train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(pure_model.unet.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/degradation_decoupling.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--unet-resume")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    main(parser.parse_args())
