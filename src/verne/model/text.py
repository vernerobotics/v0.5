# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
#
# Portions derived from Wan2.2 (https://github.com/Wan-Video/Wan2.2),
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache License 2.0. See THIRD_PARTY_NOTICES.
"""The UMT5-XXL text encoder and its tokenizer."""

import html
import math
import os

import ftfy
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from .config import BACKBONE_REPO, BACKBONE_REVISION

TEXT_ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR = "google/umt5-xxl"


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, pos_bias: torch.Tensor
    ) -> torch.Tensor:
        """x [B, L, C], mask [B, L], pos_bias [1, N, L, L]."""
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(x).view(b, -1, n, c)
        v = self.v(x).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        attn_bias += pos_bias
        if mask is not None:
            attn_bias.masked_fill_(mask.view(b, 1, 1, -1) == 0, torch.finfo(x.dtype).min)
        # T5 does not scale the scores.
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        return self.o(x.reshape(b, -1, n * c))


class T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x) * self.gate(x))


class T5RelativeEmbedding(nn.Module):
    """Bidirectional relative position bias, per head."""

    def __init__(self, num_buckets: int, num_heads: int, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, length: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(length, device=device).unsqueeze(0) - torch.arange(
            length, device=device
        ).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        return self.embedding(rel_pos).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Block(nn.Module):
    def __init__(self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int, num_buckets: int):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        pos_bias = self.pos_embedding(x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask, pos_bias))
        return fp16_clamp(x + self.ffn(self.norm2(x)))


class UMT5Encoder(nn.Module):
    """The UMT5 encoder stack. The default sizes are UMT5-XXL."""

    def __init__(
        self,
        vocab: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [T5Block(dim, dim_attn, dim_ffn, num_heads, num_buckets) for _ in range(num_layers)]
        )
        self.norm = T5LayerNorm(dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class Tokenizer:
    """The UMT5 tokenizer: whitespace clean-up, then pad to `max_len` tokens."""

    def __init__(self, path: str, max_len: int):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token ids and the attention mask, both [B, max_len].

        A prompt that fills all `max_len` tokens raises, as in training: it may be cut.
        """
        tokens = self.tokenizer(
            [clean_text(prompt) for prompt in prompts],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        full = [p for p, mask in zip(prompts, tokens.attention_mask, strict=True) if mask.all()]
        if full:
            raise ValueError(
                f"{len(full)} prompt(s) fill all {self.max_len} text tokens, so they may be "
                f"cut: {full}. Give a shorter instruction."
            )
        return tokens.input_ids, tokens.attention_mask


class TextEncoder(nn.Module):
    """Prompts -> the text rows the model reads.

    The rows after each prompt's last token are set to zero, and the returned mask is all True:
    the model attends to the zero rows as well.
    """

    def __init__(self, encoder: UMT5Encoder, tokenizer: Tokenizer):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        weights_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> "TextEncoder":
        """Load UMT5-XXL and its tokenizer. Paths left as None are downloaded."""
        if weights_path is None:
            weights_path = hf_hub_download(
                BACKBONE_REPO, TEXT_ENCODER_FILE, revision=BACKBONE_REVISION
            )
        if tokenizer_path is None:
            root = snapshot_download(
                BACKBONE_REPO, revision=BACKBONE_REVISION, allow_patterns=f"{TOKENIZER_DIR}/*"
            )
            tokenizer_path = os.path.join(root, TOKENIZER_DIR)
        with torch.device("meta"):
            encoder = UMT5Encoder()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state_dict, strict=True, assign=True)
        encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
        return cls(encoder, Tokenizer(tokenizer_path, max_len))

    @torch.no_grad()
    def forward(self, prompts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return text [B, max_len, 4096] and its mask [B, max_len] on the encoder device."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        ids, mask = self.tokenizer(prompts)
        device = self.encoder.token_embedding.weight.device
        ids = ids.to(device)
        mask = mask.to(device, dtype=torch.bool)
        text = self.encoder(ids, mask)
        for row, length in enumerate(mask.gt(0).sum(dim=1).long()):
            text[row, length:] = 0
        return text, torch.ones_like(mask)
