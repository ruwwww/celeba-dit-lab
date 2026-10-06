"""Train a latent DiT with optimal-transport flow matching and classifier-free guidance."""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.dataset import CelebAHQImageDataset, CachedLatentDataset
from models import DiT, FlowMatching, euler_sample
from vae_loader import load_frozen_vae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/mnt/data/celeba_hq/data")
    parser.add_argument("--cached-latents", default="/mnt/data/Coding3/celeba-dit-lab/cached_data/celeba_train_latents_clustered.pt")
    parser.add_argument("--vae-checkpoint", default="/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt")
    parser.add_argument("--output-dir", default="outputs/flow_dit_cfg")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--class-dropout", type=float, default=0.15)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"bf16 AMP is not supported by this script on {device.type}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class ModelEMA:
    """Device-local exponential moving average of a model's parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        self.decay = decay
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.lerp_(model_value, 1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)


def _trainable_parameters(*modules: nn.Module) -> Iterable[nn.Parameter]:
    for module in modules:
        yield from (parameter for parameter in module.parameters() if parameter.requires_grad)


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.add(1).div(2).clamp(0, 1)


def _model_config(
    latent_size: int,
    hidden_size: int = 512,
    depth: int = 12,
    num_heads: int = 8,
    num_classes: int = 0,
    class_dropout: float = 0.15,
) -> dict[str, int | float]:
    return {
        "input_size": latent_size,
        "patch_size": 1,
        "in_channels": 32,
        "out_channels": 32,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "num_classes": num_classes,
        "class_dropout_prob": class_dropout,
    }


@torch.no_grad()
def generate_images(
    model: nn.Module,
    vae: nn.Module,
    *,
    num_images: int,
    latent_size: int,
    steps: int,
    cfg_scale: float = 1.0,
    labels: torch.Tensor | None = None,
    device: torch.device,
) -> torch.Tensor:
    num_classes = getattr(model, "num_classes", 0)
    if labels is None and num_classes > 0:
        labels = torch.randint(0, num_classes, (num_images,), device=device)
    latents = euler_sample(
        model,
        (num_images, 32, latent_size, latent_size),
        y=labels,
        cfg_scale=cfg_scale,
        steps=steps,
        device=device,
        dtype=torch.float32,
    )
    return vae.decode(latents).float()


@torch.no_grad()
def save_sample_grid(
    model: nn.Module,
    vae: nn.Module,
    path: Path,
    *,
    num_images: int,
    latent_size: int,
    steps: int,
    cfg_scale: float = 3.0,
    labels: torch.Tensor | None = None,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    vae.eval()
    images = generate_images(
        model,
        vae,
        num_images=num_images,
        latent_size=latent_size,
        steps=steps,
        cfg_scale=cfg_scale,
        labels=labels,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(_denormalize(images).cpu(), path, nrow=max(1, int(num_images**0.5)))
    model.train(was_training)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    model_config: dict[str, int | float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "model_config": model_config,
        },
        path,
    )


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def train(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided")
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_size = args.image_size // 16
    model_config = _model_config(
        latent_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        class_dropout=args.class_dropout,
    )
    model = DiT(**model_config).to(device)
    ema = ModelEMA(model, args.ema_decay)
    flow = FlowMatching(model, latent_shape=(32, latent_size, latent_size))
    optimizer = AdamW(
        _trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = _load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("step", 0))

    use_cache = args.cached_latents and Path(args.cached_latents).exists()
    if use_cache:
        print(f"Using cached latents dataset from {args.cached_latents}")
        dataset = CachedLatentDataset(args.cached_latents)
    else:
        print("Cached latents not found. Falling back to on-the-fly VAE encoding.")
        dataset = CelebAHQImageDataset(
            data_dir=args.data_dir,
            image_size=args.image_size,
            split="train",
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    vae = load_frozen_vae(args.vae_checkpoint, device=device)
    amp_enabled = bool(args.amp and device.type in {"cuda", "cpu"})

    # Fixed sample labels for consistent visualization over epochs
    fixed_labels = None
    if args.num_classes > 0:
        fixed_labels = torch.arange(args.sample_count, device=device) % args.num_classes

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch_data in loader:
            if use_cache:
                clean_latents, labels = batch_data
                clean_latents = clean_latents.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                images = batch_data.to(device, non_blocking=True)
                with torch.no_grad():
                    clean_latents = vae.encode(images).float()
                labels = None

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                loss = flow.loss(clean_latents, y=labels)
            loss.backward()
            optimizer.step()
            ema.update(model)
            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} loss={loss.item():.6f}",
                    flush=True,
                )
            if global_step % args.sample_every == 0:
                save_sample_grid(
                    ema.model,
                    vae,
                    output_dir / f"sample_{global_step:08d}.png",
                    num_images=args.sample_count,
                    latent_size=latent_size,
                    steps=args.sample_steps,
                    cfg_scale=args.cfg_scale,
                    labels=fixed_labels,
                    device=device,
                )
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                model_config=model_config,
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    save_checkpoint(
        output_dir / "checkpoint_last.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        epoch=epoch if "epoch" in locals() else start_epoch - 1,
        step=global_step,
        model_config=model_config,
    )


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
