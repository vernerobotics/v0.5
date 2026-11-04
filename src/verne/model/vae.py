# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
#
# Portions derived from Wan2.2 (https://github.com/Wan-Video/Wan2.2),
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache License 2.0. See THIRD_PARTY_NOTICES.
"""The encoder of the Wan2.2 video VAE, for single frames.

Every frame is encoded alone, so a frame's latent never depends on another frame. On one
frame the causal convolutions see only their last temporal tap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from .config import BACKBONE_REPO, BACKBONE_REVISION

VAE_FILE = "Wan2.2_VAE.pth"
# Pixels per latent cell, and video frames per latent frame.
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4

LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)  # fmt: skip
LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)  # fmt: skip


class CausalConv3d(nn.Conv3d):
    """A causal 3-D convolution on one frame: the frames before it are zeros."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pad_t, pad_h, pad_w = self.padding
        if self.kernel_size[0] > 1 and 2 * pad_t != self.kernel_size[0] - 1:
            raise ValueError(
                f"A causal kernel {self.kernel_size} needs time padding of half its size"
            )
        self._padding = (pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)
        self.padding = (0, 0, 0)
        self.to(memory_format=torch.channels_last_3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size[0] > 1:
            # The zero frames meet every tap but the last, so only the last tap runs.
            x = F.pad(x, list(self._padding[:4]) + [0, 0])
            x = x.contiguous(memory_format=torch.channels_last_3d)
            return F.conv3d(
                x, self.weight[:, :, -1:], self.bias, self.stride, 0, self.dilation, self.groups
            )
        x = F.pad(x, list(self._padding))
        x = x.contiguous(memory_format=torch.channels_last_3d)
        return super().forward(x)


class RMSNorm(nn.Module):
    """RMS norm over the channel axis."""

    def __init__(self, dim: int, images: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            # Holds no weights; it keeps the checkpoint key of the next layer at index 6.
            nn.Identity(),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over the pixels of each frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t) + identity


class Downsample(nn.Module):
    """Halves the height and the width of each frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t)


class AvgDown3D(nn.Module):
    """Average-pool shortcut: folds the time and space factors into channels, then averages."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("The folded channels must divide by `out_channels`")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(B, C, T // ft, ft, H // fs, fs, W // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // ft, H // fs, W // fs)
        x = x.view(B, self.out_channels, self.group_size, T // ft, H // fs, W // fs)
        return x.mean(dim=2)


class DownBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_res_blocks: int, temporal: bool, spatial: bool
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, factor_t=2 if temporal else 1, factor_s=2 if spatial else 1
        )
        layers: list[nn.Module] = []
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if spatial:
            layers.append(Downsample(out_dim))
        self.downsamples = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsamples(x) + self.avg_shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True),
    ):
        super().__init__()
        dims = [dim * u for u in (1, *dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        blocks = []
        for i, (in_dim, block_dim) in enumerate(zip(dims[:-1], dims[1:])):
            temporal = temporal_downsample[i] if i < len(temporal_downsample) else False
            spatial = i != len(dim_mult) - 1
            blocks.append(DownBlock(in_dim, block_dim, num_res_blocks, temporal, spatial))
        self.downsamples = nn.Sequential(*blocks)
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1]),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1]),
        )
        self.head = nn.Sequential(
            RMSNorm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], out_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class VideoEncoder(nn.Module):
    """Wan2.2 VAE encoder: frames in [-1, 1] -> normalized latents, 16x smaller in space."""

    def __init__(self, dim: int = 160, z_dim: int = 48):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(dim, z_dim * 2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        mean = torch.tensor(LATENT_MEAN, device="cpu")
        inv_std = 1.0 / torch.tensor(LATENT_STD, device="cpu")
        self.register_buffer("latent_mean", mean, persistent=False)
        self.register_buffer("latent_inv_std", inv_std, persistent=False)

    @classmethod
    def from_pretrained(cls, path: str | None = None) -> "VideoEncoder":
        """Load the encoder from `Wan2.2_VAE.pth`, downloaded when `path` is None."""
        if path is None:
            path = hf_hub_download(BACKBONE_REPO, VAE_FILE, revision=BACKBONE_REVISION)
        encoder = cls()
        encoder.load_vae_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return encoder.eval().requires_grad_(False)

    def load_vae_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the encoder weights of a full Wan2.2 VAE state dict.

        The decoder and the temporal downsample convolutions do not run on single frames, so
        their weights are skipped. Any other key must match.
        """
        kept = {}
        for key, value in state_dict.items():
            if key.startswith(("decoder.", "conv2.")) or ".time_conv." in key:
                continue
            if not key.startswith(("encoder.", "conv1.")):
                raise KeyError(f"Unexpected VAE key `{key}`")
            kept[key] = value
        self.load_state_dict(kept, strict=True)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """[B, 3, T, H, W] video in [-1, 1] -> [B, z_dim, T, H / 16, W / 16], frame by frame."""
        batch = video.shape[0]
        x = rearrange(video, "b c t h w -> (b t) c 1 h w")
        x = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2)
        mu = self.conv1(self.encoder(x)).chunk(2, dim=1)[0]
        mean = self.latent_mean.to(dtype=mu.dtype).view(1, self.z_dim, 1, 1, 1)
        inv_std = self.latent_inv_std.to(dtype=mu.dtype).view(1, self.z_dim, 1, 1, 1)
        mu = (mu - mean) * inv_std
        return rearrange(mu, "(b t) c 1 h w -> b c t h w", b=batch)
