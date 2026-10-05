"""Dataset wrapper for CelebA-HQ latents or on-the-fly caching."""
import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CelebAHQImageDataset(Dataset):
    def __init__(self, data_dir="/mnt/data/celeba_hq/data", split="train", image_size=256):
        super().__init__()
        self.data_dir = Path(data_dir)
        # All images 00000.jpg to 29999.jpg
        all_files = sorted(glob.glob(str(self.data_dir / "*.jpg")))
        if not all_files:
            raise RuntimeError(f"No jpg images found in {data_dir}")
        
        # Split: first 28,000 train, last 2,000 val
        if split == "train":
            self.files = all_files[:28000]
        else:
            self.files = all_files[28000:]
            
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
