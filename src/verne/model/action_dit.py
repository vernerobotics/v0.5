# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
#
# Portions derived from Wan2.2 (https://github.com/Wan-Video/Wan2.2),
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache License 2.0. See THIRD_PARTY_NOTICES.
"""The action transformer: a narrow transformer with the attention heads of the video one."""

import torch
import torch.nn as nn

from .config import ModelConfig
from .wan_dit import (
    ROPE_LENGTH,
    Block,
    embedding_mlp,
    rope_cos_sin,
    sinusoidal_embedding_1d,
    time_mlp,
)


class ActionTransformer(nn.Module):
    """Embeds a noisy action chunk and its flow time, and decodes the velocity."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.action_hidden_dim
        self.dim = dim
        self.freq_dim = config.freq_dim
        self.num_heads = config.num_heads
        self.action_embedding = nn.Linear(config.action_dim, dim)
        self.text_embedding = embedding_mlp(config.text_dim, dim)
        self.time_embedding = time_mlp(config.freq_dim, dim)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                Block(dim, config.action_ffn_dim, config.num_heads, config.head_dim, config.eps)
                for _ in config.paired_video_blocks
            ]
        )
        self.head = nn.Linear(dim, config.action_dim)
        # Kept out of the module state so that a dtype cast leaves it in float32. `_apply`
        # moves it with the weights.
        positions = torch.arange(ROPE_LENGTH, dtype=torch.float64, device="cpu")
        self.rope = rope_cos_sin(config.head_dim, positions)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.rope = self.rope.to(self.action_embedding.weight.device)
        return self

    def time_modulation(self, timestep: torch.Tensor, batch: int) -> torch.Tensor:
        """AdaLN input of the flow time.

        `timestep` is [1] or [B] for one time per chunk, which gives [B, 6, D], or [B, T] for one
        time per chunk position, which gives [B, T, 6, D].
        """
        if timestep.ndim == 2:
            length = timestep.shape[1]
            emb = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1))
            t = self.time_embedding(emb).reshape(batch, length, self.dim)
            return self.time_projection(t).unflatten(2, (6, self.dim))
        if timestep.shape[0] == 1 and batch > 1:
            timestep = timestep.expand(batch)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        return self.time_projection(t).unflatten(1, (6, self.dim))

    def chunk_rope(self, length: int, device: torch.device) -> torch.Tensor:
        """The 1-D RoPE table of chunk positions 0..length-1: [T, 1, 2D]."""
        return self.rope[:length].view(length, 1, -1).to(device)
