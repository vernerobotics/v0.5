# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Past-frame memory: past frame latents packed into a short token stream.

The memory holds `num_past_frames` latent frames, oldest first. The schedule packs them into
slots: recent frames keep fine patches, older frames share coarse patches across time and
space. Each slot embeds with its own patch convolution, sits at its own place on the RoPE
grid, and carries one learned attention bias.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .config import ModelConfig
from .vae import TEMPORAL_COMPRESSION
from .wan_dit import rope_cos_sin, rope_parts


def pack_slots(
    num_past_frames: int, schedule: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int, int], ...]:
    """The memory slots, oldest first: (time kernel, space kernel, first frame, end frame).

    `schedule` lists (frames, time kernel, space kernel), newest frames first.
    """
    slots = []
    index = 0
    for frames, t_kernel, s_kernel in reversed(schedule):
        for _ in range(frames // t_kernel):
            slots.append((t_kernel, s_kernel, index, index + t_kernel))
            index += t_kernel
    if index != num_past_frames:
        raise ValueError(f"The schedule packs {index} frames, expected {num_past_frames}")
    return tuple(slots)


def kernel_key(t_kernel: int, s_kernel: int) -> str:
    return f"{t_kernel}x{s_kernel}"


class PastFrameMemory(nn.Module):
    """Packs past latents into tokens, their RoPE table and the slot sizes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_past_frames = config.num_past_frames
        self.frame_stride = config.past_frame_stride
        self.head_dim = config.head_dim
        self.patch_size = config.patch_size
        self.slots = pack_slots(config.num_past_frames, config.memory_schedule)
        kernels = sorted({(t, s) for t, s, _, _ in self.slots if (t, s) != (1, 1)})
        patch_t, patch_h, patch_w = config.patch_size
        self.patch_embed = nn.ModuleDict(
            {
                kernel_key(t, s): nn.Conv3d(
                    config.latent_channels,
                    config.video_dim,
                    kernel_size=(t * patch_t, s * patch_h, s * patch_w),
                    stride=(t * patch_t, s * patch_h, s * patch_w),
                )
                for t, s in kernels
            }
        )
        # One attention logit bias per slot, added on the slot's key columns.
        self.slot_bias = nn.Parameter(torch.zeros(len(self.slots)))
        # Added to a slot's tokens in proportion to the fraction of its frames that are padding.
        self.pad_embedding = nn.Parameter(torch.zeros(config.video_dim))

    def check_grid(self, height: int, width: int) -> None:
        """Refuse a latent grid that a packed patch would not cover whole."""
        for _, s_kernel, _, _ in self.slots:
            stride = (s_kernel * self.patch_size[1], s_kernel * self.patch_size[2])
            if height % stride[0] or width % stride[1]:
                raise ValueError(
                    f"The {height}x{width} latent grid does not divide by the {stride} memory patch"
                )

    def forward(
        self,
        past_latents: torch.Tensor,
        past_padded: torch.Tensor,
        frame_patch_embedding: nn.Conv3d,
    ) -> torch.Tensor:
        """Pack [B, C, K, h, w] latents, oldest first, with their [B, K] padding flags, into
        tokens [B, N, D]. A (1, 1) slot uses `frame_patch_embedding`, the video patch layer.
        """
        pad_fraction = past_padded.to(device=past_latents.device, dtype=past_latents.dtype)
        slot_tokens = []
        for t_kernel, s_kernel, lo, hi in self.slots:
            frames = past_latents[:, :, lo:hi]
            if (t_kernel, s_kernel) == (1, 1):
                embedded = frame_patch_embedding(frames)
            else:
                embedded = self.patch_embed[kernel_key(t_kernel, s_kernel)](frames)
            tokens = rearrange(embedded, "b c f h w -> b (f h w) c")
            slot_pad = pad_fraction[:, lo:hi].mean(dim=1)
            slot_tokens.append(
                tokens + slot_pad.view(-1, 1, 1) * self.pad_embedding.to(tokens.dtype)
            )
        return torch.cat(slot_tokens, dim=1)

    def rope(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The RoPE table of the memory tokens of an h x w latent grid, [N, 1, 2 * head_dim],
        and the token count of each slot."""
        f_dim, h_dim, w_dim = rope_parts(self.head_dim)
        slot_freqs, slot_sizes = [], []
        for t_kernel, s_kernel, lo, hi in self.slots:
            rows = height // (s_kernel * self.patch_size[1])
            cols = width // (s_kernel * self.patch_size[2])
            slot_sizes.append(rows * cols)

            # The slot sits at the mean time of its frames, in latent frames before the present,
            # and each packed token at the centre of the patches it covers.
            frame_pos = (
                -(self.num_past_frames - 0.5 * (lo + hi - 1))
                * self.frame_stride
                / float(TEMPORAL_COMPRESSION)
            )
            offset = 0.5 * (s_kernel - 1)
            f_pos = torch.full((1,), frame_pos, dtype=torch.float64, device=device)
            h_pos = s_kernel * torch.arange(rows, dtype=torch.float64, device=device) + offset
            w_pos = s_kernel * torch.arange(cols, dtype=torch.float64, device=device) + offset
            grid = (1, rows, cols)
            slot_freqs.append(
                torch.cat(
                    [
                        rope_cos_sin(f_dim, f_pos).view(1, 1, 1, -1).expand(*grid, -1),
                        rope_cos_sin(h_dim, h_pos).view(1, rows, 1, -1).expand(*grid, -1),
                        rope_cos_sin(w_dim, w_pos).view(1, 1, cols, -1).expand(*grid, -1),
                    ],
                    dim=-1,
                ).reshape(1, rows * cols, 1, -1)
            )
        return torch.cat(slot_freqs, dim=1)[0], tuple(slot_sizes)
