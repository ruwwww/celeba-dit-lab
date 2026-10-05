# CelebA-HQ Latent Flow Matching DiT Lab

Testbed implementasi Flow Matching / Diffusion Transformer (DiT) pada ruang laten f16 (32 kanal) yang dihasilkan oleh pre-trained Autoencoder dari `/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt`.

## Arsitektur & Spesifikasi Model

1. **Latent Space:**
   - Input image: $256 \times 256 \times 3$
   - VAE downsampling ratio: $f16$
   - Latent shape: $16 \times 16 \times 32$ (32 channels, grid $16 \times 16 = 256$ tokens jika patch=1, atau $8 \times 8 = 64$ tokens jika patch=2).
   - Latent norm: RMS Sphere normalized ($z = \tilde{z} / \|\tilde{z}\|_{\text{RMS}}$).

2. **Flow Matching Backbone (DiT):**
   - Mengikuti prinsip arsitektur modern (JiT / DiT / Flow Matching):
   - Patch size pada laten: $p=1$ (256 tokens) atau $p=2$ (64 tokens). Untuk laten $16 \times 16$, gunakan patch size $p=1$ (256 visual tokens) atau $p=2$ ($8 \times 8 = 64$ visual tokens). Default: patch size $p=1$ (256 visual tokens, sequence length 256).
   - In channels = 32, Out channels = 32.
   - Hidden size: $D = 512$, Layers: 12, Heads: 8 (Head dimension = 64). Parameter count: ~35M–45M parameter (sangat cepat dan efisien di RTX 5060 Ti 16GB).
   - Modulation: adaLN-Zero (Adaptive LayerNorm conditioned on timestep $t$).
   - 2D Sinusoidal / RoPE position embeddings.

3. **Flow Matching Formulation:**
   - Optimal Transport (OT) conditional flow path:
     $x_t = t \cdot x_1 + (1 - t) \cdot x_0$
     di mana $x_1$ adalah clean latent dari VAE, $x_0 \sim \mathcal{N}(0, I)$ adalah initial gaussian noise, dan $t \in [0, 1]$.
   - Target velocity: $u_t = x_1 - x_0$.
   - Model memprediksi $v_\theta(x_t, t) \approx u_t$.
   - Loss function: Mean Squared Error $\mathcal{L} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$.

4. **Sampling:**
   - Euler ODE Integrator:
     Mulai dari $x_0 \sim \mathcal{N}(0, I)$ di $t=0$, integrasikan menuju $t=1$:
     $x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$.
   - Langkah sampling: 25 atau 50 steps.
   - Latent hasil generasi $x_1$ didekodekan dengan VAE decoder: $\hat{x} = \text{Decoder}(x_1)$.

## VAE Frozen Checkpoint
- Lokasi: `/mnt/data/Coding3/celeba-vae-lab/outputs/adaptive_stage/checkpoint_0035.pt`
- Kelas: `CelebAVAE` (dari `/mnt/data/Coding3/celeba-vae-lab/models/autoencoder.py`)
