"""Optimal-transport flow matching utilities and Euler sampling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _batch_timesteps(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if t.ndim == 0:
        t = t.expand(batch_size)
    elif t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.ndim != 1 or t.shape[0] != batch_size:
        raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")
    return t.to(device=device, dtype=torch.float32)


def _broadcast_timesteps(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=x.dtype).reshape(x.shape[0], *([1] * (x.ndim - 1)))


def optimal_transport_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the OT interpolation ``x_t`` and constant velocity target."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}")
    if x0.ndim < 2:
        raise ValueError("flow tensors must include a batch dimension and at least one feature dimension")
    batch_t = _batch_timesteps(t, x0.shape[0], x0.device)
    t_view = _broadcast_timesteps(batch_t, x0)
    return t_view * x1 + (1.0 - t_view) * x0, x1 - x0


def flow_matching_loss(
    model: nn.Module,
    x1: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the MSE velocity loss for an OT conditional flow path."""
    if x1.ndim < 2:
        raise ValueError("x1 must include a batch dimension and at least one feature dimension")
    if x0 is None:
        x0 = torch.randn_like(x1)
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}")
    if t is None:
        t = torch.rand(x1.shape[0], device=x1.device)
    x_t, target_velocity = optimal_transport_path(x0, x1, t)
    predicted_velocity = model(x_t, _batch_timesteps(t, x1.shape[0], x1.device))
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "flow model output must match the latent shape: "
            f"expected {tuple(target_velocity.shape)}, got {tuple(predicted_velocity.shape)}"
        )
    return F.mse_loss(predicted_velocity.float(), target_velocity.float())


@contextmanager
def _evaluation_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    shape: Sequence[int],
    *,
    steps: int = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Integrate ``dx/dt = model(x, t)`` from ``t=0`` to ``t=1``."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if len(shape) < 2 or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError(f"shape must contain positive batch and feature dimensions, got {shape}")

    model_device = _model_device(model)
    sample_device = torch.device(device) if device is not None else model_device
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
    samples = torch.randn(tuple(int(dimension) for dimension in shape), device=sample_device, dtype=dtype)
    times = torch.linspace(0.0, 1.0, steps + 1, device=sample_device, dtype=torch.float32)

    with _evaluation_mode(model):
        for index in range(steps):
            timestep = times[index].expand(samples.shape[0])
            velocity = model(samples, timestep)
            if velocity.shape != samples.shape:
                raise ValueError(
                    "flow model output must match the sample shape: "
                    f"expected {tuple(samples.shape)}, got {tuple(velocity.shape)}"
                )
            samples = samples + (times[index + 1] - times[index]).to(samples.dtype) * velocity
    return samples


class FlowMatching(nn.Module):
    """Wrap a velocity model with OT flow matching and Euler sampling."""

    def __init__(self, model: nn.Module, latent_shape: Sequence[int] = (32, 16, 16)) -> None:
        super().__init__()
        if len(latent_shape) != 3 or any(int(dimension) <= 0 for dimension in latent_shape):
            raise ValueError("latent_shape must be a positive (channels, height, width) triple")
        self.model = model
        self.latent_shape = tuple(int(dimension) for dimension in latent_shape)

    def forward(
        self,
        x1: torch.Tensor,
        t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss(x1, t=t, x0=x0)

    def loss(
        self,
        x1: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flow_matching_loss(self.model, x1, t=t, x0=x0)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        steps: int = 50,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        return euler_sample(
            self.model,
            (num_samples, *self.latent_shape),
            steps=steps,
            device=device,
            dtype=dtype,
        )


# Friendly aliases for callers that name the objective after its formulation.
OTFlowMatching = FlowMatching
euler_ode_sampler = euler_sample


__all__ = [
    "FlowMatching",
    "OTFlowMatching",
    "euler_sample",
    "euler_ode_sampler",
    "flow_matching_loss",
    "optimal_transport_path",
]
