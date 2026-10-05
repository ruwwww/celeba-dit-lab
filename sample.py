"""Generate a grid of CelebA-HQ faces from a trained latent DiT."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import DiT
from train import generate_images
from vae_loader import load_frozen_vae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Flow-matching checkpoint")
    parser.add_argument("--vae-checkpoint", default="/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt")
    parser.add_argument("--output", default="samples.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nrow", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.add(1).div(2).clamp(0, 1)


def load_flow_model(checkpoint_path: str | Path, device: torch.device) -> tuple[DiT, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config")
    if config is None:
        config = {
            "input_size": 16,
            "patch_size": 1,
            "in_channels": 32,
            "out_channels": 32,
            "hidden_size": 512,
            "depth": 12,
            "num_heads": 8,
        }
    model = DiT(**config).to(device)
    state_dict = checkpoint.get("ema", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError("checkpoint must contain either 'ema' or 'model' weights")
    model.load_state_dict(state_dict)
    model.eval()
    return model, int(config["input_size"])


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, latent_size = load_flow_model(args.checkpoint, device)
    vae = load_frozen_vae(args.vae_checkpoint, device=device)
    images = generate_images(
        model,
        vae,
        num_images=args.num_samples,
        latent_size=latent_size,
        steps=args.steps,
        device=device,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrow = args.nrow if args.nrow is not None else max(1, int(args.num_samples**0.5))
    save_image(_denormalize(images).cpu(), output_path, nrow=nrow)
    print(f"saved {args.num_samples} samples to {output_path}")


def main(argv: list[str] | None = None) -> None:
    sample(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
