#!/usr/bin/env python3
"""Train the Multimodal Degradation-Aware (MDA) module."""

import csv
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dus_difuse.dataset import FusionTrainingDataset
from dus_difuse.model import MultimodalDegradationAwareEncoder
from scripts._common import project_path


def parse_labels(parameters):
    """Convert the default DataLoader's metadata batch to loss-ready tensors."""
    class_labels = parameters["type"].reshape(-1).long()
    severity_labels = parameters["params"].reshape(-1, 1).float()
    return class_labels, severity_labels


def collate_clip_batch(samples):
    """Collate only fixed-size CLIP tensors and labels from variable-size pairs."""
    batch = {
        "vi_lq_clip": torch.stack([sample["vi_lq_clip"] for sample in samples]),
        "ir_lq_clip": torch.stack([sample["ir_lq_clip"] for sample in samples]),
    }
    for modality in ("vi_lq_params", "ir_lq_params"):
        batch[modality] = {
            "type": torch.as_tensor(
                [sample[modality]["type"] for sample in samples], dtype=torch.long
            ),
            "params": torch.as_tensor(
                [sample[modality]["params"] for sample in samples], dtype=torch.float32
            ),
        }
    return batch


def resolve_paths(cfg):
    for key in ("dataset_root", "exp_dir", "init_path", "resume", "motion_kernel_path"):
        if cfg.train.get(key) is not None:
            cfg.train[key] = project_path(cfg.train[key])
    return cfg


def optimizer_for(model, cfg):
    backbone = []
    heads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("classifier_head.", "regression_head.")):
            heads.append(parameter)
        else:
            backbone.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg.train.learning_rate_backbone},
            {"params": heads, "lr": cfg.train.learning_rate_head},
        ],
        weight_decay=cfg.train.weight_decay,
    )


def checkpoint_payload(model, optimizer, epoch, best_loss, cfg):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def save_checkpoint(path, model, optimizer, epoch, best_loss, cfg):
    """Write through a temporary file so an interrupted save keeps the old file."""
    temporary_path = f"{path}.tmp"
    torch.save(
        checkpoint_payload(model, optimizer, epoch, best_loss, cfg),
        temporary_path,
    )
    os.replace(temporary_path, path)


def save_mda_weights(path, model):
    """Export only the visual encoder used to produce MDA conditions."""
    visual_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("model.visual.")
    }
    if not visual_state:
        raise RuntimeError("No MDA visual encoder parameters were found for export")
    temporary_path = f"{path}.tmp"
    torch.save(visual_state, temporary_path)
    os.replace(temporary_path, path)


def main(args):
    accelerator = Accelerator(split_batches=True)
    cfg = OmegaConf.load(project_path(args.config))
    if args.data_root:
        cfg.train.dataset_root = args.data_root
    if args.output_dir:
        cfg.train.exp_dir = args.output_dir
    if args.init_path:
        cfg.train.init_path = args.init_path
    if args.resume:
        cfg.train.resume = args.resume
    resolve_paths(cfg)
    set_seed(cfg.train.seed, device_specific=True)

    checkpoint_dir = os.path.join(cfg.train.exp_dir, "checkpoints")
    log_path = os.path.join(cfg.train.exp_dir, "training_metrics.csv")
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(cfg.train.exp_dir)

    dataset = FusionTrainingDataset(
        cfg.train.dataset_root,
        crop_size=cfg.train.crop_size,
        return_params=True,
        motion_kernel_path=cfg.train.motion_kernel_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=cfg.train.drop_last,
        collate_fn=collate_clip_batch,
    )
    if not len(loader):
        raise RuntimeError("No training batches; reduce batch_size or disable drop_last")

    init_path = None if cfg.train.resume else cfg.train.init_path
    model = MultimodalDegradationAwareEncoder(
        embed_dim=cfg.model.embed_dim,
        vision_cfg=cfg.model.vision_cfg,
        text_cfg=cfg.model.text_cfg,
        pretrained_path=init_path,
        num_classes=cfg.model.num_classes,
    )
    optimizer = optimizer_for(model, cfg)
    start_epoch = 0
    best_loss = float("inf")
    if cfg.train.resume:
        resume = torch.load(
            cfg.train.resume, map_location="cpu", weights_only=False, mmap=True
        )
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"])
        best_loss = float(resume.get("best_loss", resume.get("avg_loss", best_loss)))

    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.MSELoss()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process and (start_epoch == 0 or not os.path.isfile(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(
                ["epoch", "loss", "classification_loss", "regression_loss", "accuracy", "mae"]
            )

    global_step = start_epoch * len(loader)
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        sums = torch.zeros(6, device=accelerator.device)
        progress = tqdm(
            loader,
            disable=not accelerator.is_main_process,
            desc=f"epoch {epoch + 1}/{cfg.train.epochs}",
            unit="batch",
        )
        for batch in progress:
            images = torch.cat((batch["vi_lq_clip"], batch["ir_lq_clip"]), dim=0).float()
            visible_class, visible_severity = parse_labels(batch["vi_lq_params"])
            infrared_class, infrared_severity = parse_labels(batch["ir_lq_params"])
            class_labels = torch.cat((visible_class, infrared_class), dim=0).to(images.device)
            severity_labels = torch.cat((visible_severity, infrared_severity), dim=0).to(images.device)
            if class_labels.min() < 0 or class_labels.max() >= cfg.model.num_classes:
                raise ValueError(
                    f"Class labels must be in [0, {cfg.model.num_classes - 1}], got "
                    f"[{class_labels.min().item()}, {class_labels.max().item()}]"
                )

            class_prediction, severity_prediction = model(images)
            loss_class = classification_loss(class_prediction, class_labels)
            loss_regression = regression_loss(severity_prediction, severity_labels)
            loss = loss_class + cfg.train.regression_weight * loss_regression

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if cfg.train.max_grad_norm:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()

            batch_size = class_labels.numel()
            correct = (class_prediction.argmax(dim=1) == class_labels).sum()
            absolute_error = (severity_prediction.detach() - severity_labels).abs().sum()
            sums += torch.stack(
                (
                    loss.detach() * batch_size,
                    loss_class.detach() * batch_size,
                    loss_regression.detach() * batch_size,
                    correct,
                    absolute_error,
                    torch.as_tensor(batch_size, device=images.device),
                )
            )
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if writer and global_step % cfg.train.log_every == 0:
                writer.add_scalar("batch/loss", loss.item(), global_step)

        sums = accelerator.reduce(sums, reduction="sum")
        count = sums[5].item()
        metrics = {
            "loss": sums[0].item() / count,
            "classification_loss": sums[1].item() / count,
            "regression_loss": sums[2].item() / count,
            "accuracy": sums[3].item() / count,
            "mae": sums[4].item() / count,
        }
        improved = metrics["loss"] < best_loss
        best_loss = min(best_loss, metrics["loss"])

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with open(log_path, "a", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow([epoch + 1, *metrics.values()])
            for name, value in metrics.items():
                writer.add_scalar(f"epoch/{name}", value, epoch + 1)
            if improved:
                save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pth"),
                    unwrapped_model, optimizer, epoch + 1, best_loss, cfg,
                )
                save_mda_weights(
                    os.path.join(checkpoint_dir, "mda.pth"), unwrapped_model
                )
            if (epoch + 1) % cfg.train.save_every == 0:
                save_checkpoint(
                    os.path.join(checkpoint_dir, "latest.pth"),
                    unwrapped_model, optimizer, epoch + 1, best_loss, cfg,
                )

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/mda.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--init-path")
    parser.add_argument("--resume")
    main(parser.parse_args())
