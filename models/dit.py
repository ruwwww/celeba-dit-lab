"""Diffusion Transformer backbone for 16x16 VAE latents.

The model follows the DiT adaLN-Zero design: timestep embeddings produce
shift, scale, and residual-gate values for each transformer block, while the
final projection is zero initialized so the network starts as a zero velocity
predictor.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


def _as_pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("expected an integer or a pair of integers")
    return int(value[0]), int(value[1])


def timestep_embedding(timesteps: torch.Tensor, dimension: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal embeddings for scalar timesteps in ``[0, 1]``."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape (N,), got {tuple(timesteps.shape)}")
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")

    half = dimension // 2
    if half == 0:
        return timesteps[:, None]
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    arguments = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def _sincos_2d_position_embedding(height: int, width: int, dimension: int) -> torch.Tensor:
    """Return a ``(height * width, dimension)`` fixed 2-D sin/cos embedding."""
    if dimension % 4:
        raise ValueError("2-D sinusoidal position embeddings require dimension divisible by 4")

    quarter = dimension // 4
    omega = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10_000 ** (omega / max(quarter - 1, 1)))
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    y_embedding = torch.einsum("m,d->md", y, omega)
    x_embedding = torch.einsum("m,d->md", x, omega)
    y_embedding = torch.cat((torch.sin(y_embedding), torch.cos(y_embedding)), dim=-1)
    x_embedding = torch.cat((torch.sin(x_embedding), torch.cos(x_embedding)), dim=-1)
    y_embedding = y_embedding[:, None, :].expand(height, width, -1)
    x_embedding = x_embedding[None, :, :].expand(height, width, -1)
    return torch.cat((y_embedding, x_embedding), dim=-1).reshape(height * width, dimension)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DiTBlock(nn.Module):
    """Transformer block with timestep-conditioned adaLN-Zero modulation."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, timestep_embedding_value: torch.Tensor) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(timestep_embedding_value).chunk(6, dim=-1)
        )
        x = x + gate_attention.unsqueeze(1) * self.attention(
            _modulate(self.norm1(x), shift_attention, scale_attention)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Conditioned output projection from tokens back to latent patches."""

    def __init__(self, hidden_size: int, patch_output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, patch_output_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, timestep_embedding_value: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep_embedding_value).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Diffusion Transformer predicting velocity fields over VAE latents."""

    def __init__(
        self,
        input_size: int | Tuple[int, int] = 16,
        patch_size: int = 1,
        in_channels: int = 32,
        out_channels: int | None = None,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        input_height, input_width = _as_pair(input_size)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if input_height % patch_size or input_width % patch_size:
            raise ValueError("input dimensions must be divisible by patch_size")
        if min(in_channels, hidden_size, depth, num_heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.input_size = (input_height, input_width)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (input_height // patch_size) * (input_width // patch_size)

        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        position_embedding = _sincos_2d_position_embedding(
            input_height // patch_size,
            input_width // patch_size,
            hidden_size,
        )
        self.register_buffer("pos_embed", position_embedding.unsqueeze(0), persistent=False)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_size, patch_size * patch_size * self.out_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.zeros_(self.t_embedder[0].bias)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)
        nn.init.zeros_(self.t_embedder[2].bias)
        zero_initialized = {
            block.adaLN_modulation[-1] for block in self.blocks
        }
        zero_initialized.update({
            self.final_layer.linear,
            self.final_layer.adaLN_modulation[-1],
        })
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in {
                self.t_embedder[0],
                self.t_embedder[2],
            } | zero_initialized:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _position_embedding(self, height: int, width: int, x: torch.Tensor) -> torch.Tensor:
        if (height, width) == self.input_size:
            return self.pos_embed.to(device=x.device, dtype=x.dtype)
        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        position_embedding = _sincos_2d_position_embedding(
            patch_height,
            patch_width,
            self.hidden_size,
        )
        return position_embedding.to(device=x.device, dtype=x.dtype).unsqueeze(0)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected latent input (N, {self.in_channels}, H, W), got {tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("latent height and width must be divisible by patch_size")
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.shape[0])
        elif timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                f"timesteps must have shape ({x.shape[0]},), got {tuple(timesteps.shape)}"
            )

        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        tokens = self.x_embedder(x).flatten(2).transpose(1, 2)
        tokens = tokens + self._position_embedding(height, width, x)
        timestep_value = self.t_embedder(timestep_embedding(timesteps, self.hidden_size).to(x.dtype))
        for block in self.blocks:
            tokens = block(tokens, timestep_value)
        patches = self.final_layer(tokens, timestep_value)
        patches = patches.reshape(
            x.shape[0],
            patch_height,
            patch_width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            x.shape[0], self.out_channels, height, width
        )


__all__ = ["DiT", "DiTBlock", "FinalLayer", "timestep_embedding"]
