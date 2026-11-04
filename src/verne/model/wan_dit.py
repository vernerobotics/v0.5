# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
#
# Portions derived from Wan2.2 (https://github.com/Wan-Video/Wan2.2),
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache License 2.0. See THIRD_PARTY_NOTICES.
"""The video transformer and the building blocks both transformers share."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import ModelConfig

ROPE_LENGTH = 1024
ROPE_THETA = 10000.0


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """[N] positions -> [N, dim] cos/sin features, in the dtype of `position`."""
    half = dim // 2
    inv_freq = torch.pow(
        10000, -torch.arange(half, dtype=torch.float64, device=position.device).div(half)
    )
    sinusoid = torch.outer(position.type(torch.float64), inv_freq)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def rope_cos_sin(dim: int, positions: torch.Tensor, theta: float = ROPE_THETA) -> torch.Tensor:
    """The RoPE table of `dim` channels at `positions` [N]: [N, 2 * dim] float32.

    Each channel holds cos(angle) and a signed sin(angle): minus on an even channel, plus on
    an odd one. Channels 2i and 2i + 1 rotate together. Positions may be fractional or negative.
    """
    pairs = dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=positions.device)[:pairs].double() / dim)
    )
    angle = torch.outer(positions.double(), inv_freq)
    cos = torch.cos(angle).repeat_interleave(2, dim=-1)
    sin = torch.sin(angle).repeat_interleave(2, dim=-1)
    sign = torch.ones(2 * pairs, dtype=torch.float64, device=positions.device)
    sign[0::2] = -1.0
    return torch.stack([cos, sin * sign], dim=-1).flatten(1).float()


def rope_parts(head_dim: int) -> tuple[int, int, int]:
    """The (frame, row, column) channel counts of one head of the 3-D video RoPE."""
    return head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Rotate queries or keys [B, S, H * D] by a [S, 1, 2 * D] table, in float32."""
    dtype = x.dtype
    cos, sin = freqs.unflatten(-1, (-1, 2)).unbind(-1)
    partner = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2).unflatten(-1, (num_heads, -1))
    heads = x.unflatten(-1, (num_heads, -1))
    return torch.addcmul(heads * cos, partner, sin).flatten(-2).to(dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention on [B, S, H * D] tensors."""
    q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(x, "b n s d -> b s (n d)")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def split_modulation(modulation: torch.Tensor, t_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The six AdaLN terms of a block. `t_mod` is [B, 6, D], or [B, S, 6, D] per token."""
    per_token = t_mod.ndim == 4
    chunks = (modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        6, dim=2 if per_token else 1
    )
    if per_token:
        chunks = tuple(chunk.squeeze(2) for chunk in chunks)
    return chunks


def embedding_mlp(in_dim: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))


def time_mlp(freq_dim: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner)
        self.k = nn.Linear(dim, inner)
        self.v = nn.Linear(dim, inner)
        self.o = nn.Linear(inner, dim)
        # The norm spans all heads together.
        self.norm_q = RMSNorm(inner, eps)
        self.norm_k = RMSNorm(inner, eps)


class KeyValue(nn.Module):
    """The key and value projections of a self-attention layer."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = num_heads * head_dim
        self.k = nn.Linear(dim, inner)
        self.v = nn.Linear(dim, inner)
        self.norm_k = RMSNorm(inner, eps)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, inner)
        self.k = nn.Linear(dim, inner)
        self.v = nn.Linear(dim, inner)
        self.o = nn.Linear(inner, dim)
        self.norm_q = RMSNorm(inner, eps)
        self.norm_k = RMSNorm(inner, eps)

    def project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The keys and values of `context` [B, L, D], each [B, L, H * D]."""
        return self.norm_k(self.k(context)), self.v(context)

    def forward(
        self,
        x: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend from `x` to the context keys and values from `project_kv`."""
        q = self.norm_q(self.q(x))
        k, v = context_kv
        return self.o(attention(q, k, v, self.num_heads, mask))


class Block(nn.Module):
    """A transformer block with AdaLN time modulation, self-attention, cross-attention and FFN.

    The self-attention runs outside the block, so two transformers can attend jointly:
    `attention_inputs` gives the queries, keys and values, and `finish` takes the result.
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = SelfAttention(dim, num_heads, head_dim, eps)
        self.cross_attn = CrossAttention(dim, num_heads, head_dim, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def attention_inputs(
        self, x: torch.Tensor, t_mod: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return q, k and v [B, S, H * D] with RoPE applied, and the state `finish` reads."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_modulation(
            self.modulation, t_mod
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self.self_attn
        k = rope_apply(attn.norm_k(attn.k(h)), freqs, self.num_heads)
        v = attn.v(h)
        q = rope_apply(attn.norm_q(attn.q(h)), freqs, self.num_heads)
        return q, k, v, (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def finish(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        state: tuple[torch.Tensor, ...],
        context_kv: tuple[torch.Tensor, torch.Tensor],
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the self-attention output, the cross-attention and the FFN to `x`."""
        gate_msa, shift_mlp, scale_mlp, gate_mlp = state
        x = x + gate_msa * self.self_attn.o(attn_out)
        x = x + self.cross_attn(self.norm3(x), context_kv, context_mask)
        return x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))


class KeyValueBlock(nn.Module):
    """The last video block an action layer reads. Only its keys and values are used."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = KeyValue(dim, num_heads, head_dim, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def key_value(
        self, x: torch.Tensor, t_mod: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa = split_modulation(self.modulation, t_mod)[:2]
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        k = rope_apply(self.self_attn.norm_k(self.self_attn.k(h)), freqs, self.num_heads)
        return k, self.self_attn.v(h)


class VideoTransformer(nn.Module):
    """The video transformer. It reads the memory and the present frame latents once per chunk.

    Video block `paired_blocks[i]` shares its attention with action layer `i`. Every block up to
    the last paired one runs; the last paired block only gives its keys and values.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.video_dim
        self.dim = dim
        self.freq_dim = config.freq_dim
        self.num_heads = config.num_heads
        self.paired_blocks = config.paired_video_blocks
        self.patch_embedding = nn.Conv3d(
            config.latent_channels, dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.text_embedding = embedding_mlp(config.text_dim, dim)
        self.time_embedding = time_mlp(config.freq_dim, dim)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        heads, head_dim, eps = config.num_heads, config.head_dim, config.eps
        full = [
            Block(dim, config.video_ffn_dim, heads, head_dim, eps)
            for _ in range(self.paired_blocks[-1])
        ]
        self.blocks = nn.ModuleList([*full, KeyValueBlock(dim, heads, head_dim, eps)])
        # Kept out of the module state so that a dtype cast leaves them in float32. `_apply`
        # moves them with the weights.
        positions = torch.arange(ROPE_LENGTH, dtype=torch.float64, device="cpu")
        self.rope = tuple(rope_cos_sin(part, positions) for part in rope_parts(config.head_dim))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.rope = tuple(table.to(self.patch_embedding.weight.device) for table in self.rope)
        return self

    def time_modulation(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Per-token AdaLN input: [B, N] timesteps -> [B, N, 6, D]."""
        batch = timesteps.shape[0]
        emb = sinusoidal_embedding_1d(self.freq_dim, timesteps.reshape(-1))
        t = self.time_embedding(emb).reshape(batch, -1, self.dim)
        return self.time_projection(t).unflatten(2, (6, self.dim))

    def frame_rope(self, frames: int, rows: int, cols: int, device: torch.device) -> torch.Tensor:
        """The RoPE table of a latent grid at frame positions 0..frames-1: [F * h * w, 1, 2D]."""
        grid = (frames, rows, cols)
        f_table, h_table, w_table = self.rope
        table = torch.cat(
            [
                f_table[:frames].view(frames, 1, 1, -1).expand(*grid, -1),
                h_table[:rows].view(1, rows, 1, -1).expand(*grid, -1),
                w_table[:cols].view(1, 1, cols, -1).expand(*grid, -1),
            ],
            dim=-1,
        )
        return table.reshape(frames * rows * cols, 1, -1).to(device)
