import importlib.util
import sys
from pathlib import Path

import torch


_VAE_PACKAGE_NAME = "_celeba_vae_models"


def _load_vae_class(vae_repo: str):
    """Load the external VAE package without colliding with this repo's models package."""
    package_path = Path(vae_repo) / "models"
    package_init = package_path / "__init__.py"
    package = sys.modules.get(_VAE_PACKAGE_NAME)
    if package is None:
        spec = importlib.util.spec_from_file_location(
            _VAE_PACKAGE_NAME,
            package_init,
            submodule_search_locations=[str(package_path)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load VAE package from {package_init}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[_VAE_PACKAGE_NAME] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(_VAE_PACKAGE_NAME, None)
            raise
    return package.CelebAVAE


def load_frozen_vae(
    checkpoint_path="/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt",
    device="cuda",
):
    vae_repo = "/mnt/data/Coding3/celeba-vae-lab"
    CelebAVAE = _load_vae_class(vae_repo)

    model = CelebAVAE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("ema", ckpt.get("model"))
    if state_dict is None:
        raise KeyError("VAE checkpoint must contain either 'ema' or 'model' weights")
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
