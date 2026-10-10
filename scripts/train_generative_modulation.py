#!/usr/bin/env python3
"""Train mask/text-conditioned Generative Modulation (Eq. 21)."""

import gc
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

from dus_difuse.dataset import GenerationDataset
from dus_difuse.model import Diffusion, DuSDiFuseModel
from dus_difuse.utils.common import instantiate_from_config, to
from scripts._common import load_state, project_path, resolve_train_paths
from scripts.generative_modulation_data import collate_generation


def main(args) -> None:
    cfg = OmegaConf.load(project_path(args.config))
    if args.data_root:
        cfg.train.image_roots, cfg.train.image_ratios = [args.data_root], [1]
    if args.label_root:
        cfg.train.label_root = args.label_root
    if args.output_dir:
        cfg.train.exp_dir = args.output_dir
    if args.controlnet_resume:
        cfg.train.resume = args.controlnet_resume
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.ckpt_every is not None:
        cfg.train.ckpt_every = args.ckpt_every
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    resolve_train_paths(cfg)

    if cfg.train.batch_size <= 0 or cfg.train.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation steps must be positive")
    if cfg.train.log_every <= 0 or cfg.train.ckpt_every <= 0:
        raise ValueError("Logging and checkpoint intervals must be positive")
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )
    if cfg.train.batch_size % cfg.train.gradient_accumulation_steps != 0:
        raise ValueError(
            "Global batch size must be divisible by gradient accumulation steps"
        )
    micro_batch_size = (
        cfg.train.batch_size // cfg.train.gradient_accumulation_steps
    )
    if micro_batch_size % accelerator.num_processes != 0:
        raise ValueError(
            "Global micro-batch size must be divisible by the process count"
        )
    set_seed(cfg.train.seed, device_specific=True)

    checkpoint_dir = os.path.join(cfg.train.exp_dir, "checkpoints")
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(cfg.train.exp_dir)

    # Construct and load one rank at a time. A complete SD v2.1 checkpoint and
    # its destination model otherwise peak in host memory on every rank at once.
    model = None
    for rank in range(accelerator.num_processes):
        if accelerator.process_index == rank:
            model = instantiate_from_config(cfg.model.dus_difuse)
            sd_state = load_state(cfg.train.sd_path)
            _, missing = model.load_pretrained_sd(sd_state)
            del sd_state
            if missing:
                raise RuntimeError(
                    "Stable Diffusion checkpoint is missing model parameters: "
                    + ", ".join(sorted(missing)[:10])
                )
            if cfg.train.resume:
                controlnet_state = load_state(cfg.train.resume)
                model.load_controlnet_from_ckpt(controlnet_state)
                del controlnet_state
            else:
                model.load_controlnet_from_unet()
            model.train().to(accelerator.device)
            gc.collect()
        accelerator.wait_for_everyone()
    if model is None:
        raise RuntimeError("Model initialization did not run on this process")

    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("controlnet.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Only ControlNet may remain trainable; unexpected parameters: "
            + ", ".join(unexpected_trainable[:10])
        )
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    optimizer = torch.optim.AdamW(model.controlnet.parameters(), lr=cfg.train.learning_rate)
    dataset = GenerationDataset(
        image_roots=cfg.train.image_roots,
        image_ratios=cfg.train.image_ratios,
        label_root=cfg.train.label_root,
        class_names=cfg.train.class_names,
        label_all_one_prob=cfg.train.label_all_one_prob,
        prompt_drop_prob=cfg.train.prompt_drop_prob,
        mask_drop_prob=cfg.train.mask_drop_prob,
        motion_kernel_path=cfg.train.motion_kernel_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=cfg.train.drop_last,
        pin_memory=True,
        collate_fn=partial(
            collate_generation,
            divisor=cfg.train.divisor,
            max_size=cfg.train.max_size,
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
        print(f"Dataset samples: {len(dataset):,}; micro-batches per pass: {len(loader):,}")
        print(f"Stable Diffusion checkpoint: {cfg.train.sd_path}")
        print(
            "ControlNet initialization: "
            + (str(cfg.train.resume) if cfg.train.resume else "frozen SD UNet")
        )
        print(
            f"Trainable ControlNet parameters: {trainable_parameters:,}; "
            f"frozen parameters: {frozen_parameters:,}"
        )
        print(
            f"Effective global batch size: {cfg.train.batch_size}; "
            f"global micro-batch size: {micro_batch_size}; "
            f"per-GPU micro-batch size: {micro_batch_size // accelerator.num_processes}; "
            f"gradient accumulation steps: {cfg.train.gradient_accumulation_steps}"
        )
        print(
            f"Training steps: {cfg.train.train_steps:,}; log every: "
            f"{cfg.train.log_every:,}; checkpoint every: {cfg.train.ckpt_every:,}"
        )

    diffusion.to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    pure_model: DuSDiFuseModel = accelerator.unwrap_model(model)
    global_step = 0
    losses = []
    optimizer.zero_grad(set_to_none=True)
    gradient_norm = None

    while global_step < cfg.train.train_steps:
        progress = tqdm(loader, disable=not accelerator.is_main_process, unit="batch")
        for batch in progress:
            with accelerator.accumulate(model):
                batch = to(batch, accelerator.device)
                with torch.no_grad():
                    target_latent = pure_model.vae_encode(batch["target"].float())
                    condition = pure_model.prepare_generation_condition(
                        batch["control"].float(), batch["mask"].float(), batch["prompt"]
                    )
                timestep = torch.randint(
                    diffusion.num_timesteps,
                    (target_latent.shape[0],),
                    device=accelerator.device,
                )
                loss, _, _ = diffusion.generative_modulation_loss(
                    model, target_latent, timestep, condition
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(
                        pure_model.controlnet.parameters(), 1.0
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.detach())
            if not accelerator.sync_gradients:
                progress.set_description(
                    f"step {global_step:07d} accumulating | L_mod {loss.item():.6f}"
                )
                continue

            global_step += 1
            progress.set_description(f"step {global_step:07d} | L_mod {loss.item():.6f}")
            if global_step % cfg.train.log_every == 0:
                mean_loss = accelerator.gather(torch.stack(losses)).mean().item()
                losses.clear()
                if writer:
                    writer.add_scalar("loss/generative_modulation", mean_loss, global_step)
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
                        pure_model.controlnet.state_dict(),
                        os.path.join(checkpoint_dir, f"{global_step:07d}.pt"),
                    )
            if global_step >= cfg.train.train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(pure_model.controlnet.state_dict(), os.path.join(checkpoint_dir, "final.pt"))
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/generative_modulation.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--label-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--controlnet-resume")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    main(parser.parse_args())
