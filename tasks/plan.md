# Implementation Plan - CelebA-HQ Latent Flow Matching DiT

## Directory Structure
- `models/dit.py`: DiT backbone dengan adaLN-Zero modulation, patch projection 1x1 atau 2x2, multi-head self-attention, dan final layer projection.
- `models/flow.py`: FlowMatching wrapper yang mengurus OT-Flow sampling ($x_t = t \cdot x_1 + (1-t) \cdot x_0$), target velocity $u_t = x_1 - x_0$, loss computation, dan Euler ODE sampler.
- `models/vae_loader.py`: Helper untuk memuat frozen CelebAVAE dari `/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt`.
- `models/__init__.py`: Exports.
- `train.py`: Training loop flow matching dengan bfloat16 AMP, AdamW (lr 2e-4), EMA, caching latent (atau on-the-fly VAE encode), sample generation berkala (Euler ODE -> VAE Decode -> simpan PNG), dan checkpointing.
- `sample.py`: CLI script untuk generate N gambar wajah acak menggunakan ODE solver dan VAE decoder.
- `tests/test_flow_dit.py`: Pytest suite untuk memvalidasi tensor shapes, forward flow matching loss, dan Euler sampling step.
