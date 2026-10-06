import torch
from torch import nn

from models.dit import DiT
from models.flow import FlowMatching, euler_sample


def make_tiny_dit() -> DiT:
    return DiT(
        input_size=16,
        patch_size=1,
        in_channels=32,
        hidden_size=64,
        depth=2,
        num_heads=4,
    )


def test_dit_forward_preserves_latent_shape():
    model = make_tiny_dit()
    latents = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.0, 0.75])

    prediction = model(latents, timesteps)

    assert prediction.shape == latents.shape
    assert prediction.dtype == latents.dtype


def test_flow_matching_loss_uses_ot_velocity_target():
    class ZeroVelocity(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    x1 = torch.tensor([[[[2.0]]], [[[4.0]]]])
    x0 = torch.tensor([[[[1.0]]], [[[1.0]]]])
    flow = FlowMatching(ZeroVelocity())

    loss = flow.loss(x1, t=torch.tensor([0.25, 0.75]), x0=x0)

    assert torch.isclose(loss, torch.tensor(5.0)).item()


def test_flow_matching_backward_populates_dit_gradients():
    model = make_tiny_dit()
    flow = FlowMatching(model)
    clean_latents = torch.randn(2, 32, 16, 16)

    loss = flow.loss(clean_latents)
    loss.backward()

    assert loss.ndim == 0
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_five_step_euler_sampling_returns_latent_batch():
    class ConstantVelocity(nn.Module):
        def forward(self, x, t):
            return torch.ones_like(x)

    model = ConstantVelocity()
    samples = euler_sample(model, shape=(3, 32, 16, 16), steps=5)

    assert samples.shape == (3, 32, 16, 16)
    assert torch.isfinite(samples).all()


def test_dit_with_cfg_and_labels():
    model = DiT(
        input_size=16,
        patch_size=1,
        in_channels=32,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=64,
        class_dropout_prob=0.15,
    )
    latents = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.2, 0.8])
    labels = torch.tensor([5, 12])

    pred = model(latents, timesteps, y=labels)
    assert pred.shape == latents.shape

    samples = euler_sample(
        model,
        shape=(2, 32, 16, 16),
        y=labels,
        cfg_scale=2.5,
        steps=5,
    )
    assert samples.shape == (2, 32, 16, 16)
