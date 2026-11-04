# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Flow-matching sampler: the shifted sigma schedule and the Euler step."""

import torch


def sigma_schedule(
    num_steps: int,
    shift: float,
    num_train_timesteps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model timesteps [num_steps] and sigma steps [num_steps], from sigma 1 down to 0.

    sigma = shift * u / (1 + (shift - 1) * u) on a uniform grid of u. The timestep the model
    reads is sigma * num_train_timesteps; each delta is the next sigma minus this one.
    """
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}")
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    sigma = shift * u / (1.0 + (shift - 1.0) * u)
    now, nxt = sigma[:-1], sigma[1:]
    return (now * float(num_train_timesteps)).to(dtype), (nxt - now).to(dtype)


def euler_step(sample: torch.Tensor, velocity: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """x + delta * v, with `delta` a scalar tensor."""
    return sample + velocity * delta.to(sample.device, dtype=sample.dtype)
