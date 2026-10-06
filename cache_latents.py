import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import numpy as np

from data.dataset import CelebAHQImageDataset
from vae_loader import load_frozen_vae

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("/mnt/data/Coding3/celeba-dit-lab/cached_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Frozen VAE...")
    vae = load_frozen_vae(device=device)
    
    print("Loading DINOv2-small for semantic feature clustering...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    
    # Process both train (28,000) and val (2,000)
    for split in ["train", "val"]:
        cache_file = output_dir / f"{split}_data.pt"
        if cache_file.exists():
            print(f"Cache {cache_file} already exists, skipping.")
            continue
            
        dataset = CelebAHQImageDataset(split=split, image_size=256)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        all_latents = []
        all_dino_feats = []
        
        print(f"Extracting latents and DINOv2 features for {split} ({len(dataset)} images)...")
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(device)
                
                # 1. VAE Latents (bfloat16)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    latents = vae.encode(batch).to(torch.bfloat16)
                all_latents.append(latents.cpu())
                
                # 2. DINOv2 Features (resize to 224x224 for standard ViT)
                batch_224 = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = dino(batch_224)
                all_dino_feats.append(feats.cpu())
                
        all_latents = torch.cat(all_latents, dim=0)
        all_dino_feats = torch.cat(all_dino_feats, dim=0).float().numpy()
        
        torch.save({
            "latents": all_latents,
            "dino_feats": all_dino_feats
        }, cache_file)
        print(f"Saved {cache_file}: latents shape {all_latents.shape}")

    # Cluster train features into 64 semantic pseudo-classes
    train_cache = torch.load(output_dir / "train_data.pt", weights_only=False)
    val_cache = torch.load(output_dir / "val_data.pt", weights_only=False)
    
    print("Fitting MiniBatchKMeans (K=64) on DINOv2 features...")
    kmeans = MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=1024, max_iter=100)
    train_labels = kmeans.fit_predict(train_cache["dino_feats"])
    val_labels = kmeans.predict(val_cache["dino_feats"])
    
    # Save final compact cached files
    torch.save({
        "latents": train_cache["latents"],
        "labels": torch.tensor(train_labels, dtype=torch.long)
    }, output_dir / "celeba_train_latents_clustered.pt")
    
    torch.save({
        "latents": val_cache["latents"],
        "labels": torch.tensor(val_labels, dtype=torch.long)
    }, output_dir / "celeba_val_latents_clustered.pt")
    
    print("Successfully cached clustered datasets!")
    print(f"Train latents: {train_cache['latents'].shape}, labels: {train_labels.shape}")

if __name__ == "__main__":
    main()
