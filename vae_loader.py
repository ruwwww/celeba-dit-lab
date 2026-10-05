import sys
from pathlib import Path
import torch

def load_frozen_vae(checkpoint_path="/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt", device="cuda"):
    vae_repo = "/mnt/data/Coding3/celeba-vae-lab"
    if vae_repo not in sys.path:
        sys.path.insert(0, vae_repo)
        
    from models.autoencoder import CelebAVAE
    
    model = CelebAVAE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("ema", ckpt.get("model"))
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
