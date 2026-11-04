# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Mixture of transformers: the video and the action transformer attend jointly, layer by layer.

The video tokens never read the action tokens, so the video transformer runs once per chunk
and caches its keys and values. Each denoising step then runs only the action transformer,
whose queries read the cached video keys and values next to the action ones.
"""

import torch

from .action_dit import ActionTransformer
from .wan_dit import VideoTransformer, attention


def joint_attention_mask(
    slot_sizes: tuple[int, ...],
    frame_tokens: int,
    action_tokens: int,
    slot_bias: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The additive mask over [memory slots, present frame, actions], [S, S].

    Memory slot i reads slots 0..i. The present frame reads the memory and itself. The actions
    read every video token and every action token. Each memory key column carries its slot's
    bias; a blocked entry is -inf.
    """
    past = sum(slot_sizes)
    video = past + frame_tokens
    total = video + action_tokens
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    end = 0
    for size in slot_sizes:
        end += size
        allowed[end - size : end, :end] = True
    allowed[past:video, :video] = True
    allowed[video:, :] = True
    parts = [slot_bias[index].reshape(1).expand(size) for index, size in enumerate(slot_sizes)]
    parts.append(torch.zeros(total - past, dtype=slot_bias.dtype, device=slot_bias.device))
    bias = torch.cat(parts).to(device=device, dtype=dtype)
    blocked = torch.full((), float("-inf"), dtype=dtype, device=device)
    return torch.where(allowed, bias, blocked)


def project_context(
    video: VideoTransformer,
    action: ActionTransformer,
    text: torch.Tensor,
    state_row: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """The context keys and values that change each chunk.

    `text` [B, L, text_dim] and the robot state row [B, 1, text_dim]. Returns the keys and
    values of the state row in each full video block, [blocks, 2, B, 1, H * D], and those of
    the text and state row in each action block.
    """
    video_row = video.text_embedding(state_row)
    row_kv = [t for block in video.blocks[:-1] for t in block.cross_attn.project_kv(video_row)]
    action_context = action.text_embedding(torch.cat([text, state_row], dim=1))
    action_kv = [block.cross_attn.project_kv(action_context) for block in action.blocks]
    return torch.stack(row_kv).unflatten(0, (-1, 2)), action_kv


def run_video_blocks(
    video: VideoTransformer,
    tokens: torch.Tensor,
    freqs: torch.Tensor,
    t_mod: torch.Tensor,
    context_kv: list[tuple[torch.Tensor, torch.Tensor]],
    mask: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run the video blocks once and return the keys and values of each paired block.

    `tokens` [B, Sv, D], `freqs` [Sv, 1, 2 * head_dim], `t_mod` [B, Sv, 6, D], `context_kv`
    the text keys and values of each full block, `mask` the [Sv, Sv] self-attention mask.
    """
    paired = set(video.paired_blocks)
    cache = []
    x = tokens
    for index, (block, kv) in enumerate(zip(video.blocks[:-1], context_kv, strict=True)):
        q, k, v, state = block.attention_inputs(x, t_mod, freqs)
        if index in paired:
            cache.append((k, v))
        x = block.finish(x, attention(q, k, v, video.num_heads, mask), state, kv, None)
    cache.append(video.blocks[-1].key_value(x, t_mod, freqs))
    return cache


def run_action_blocks(
    action: ActionTransformer,
    tokens: torch.Tensor,
    freqs: torch.Tensor,
    t_mod: torch.Tensor,
    context_kv: list[tuple[torch.Tensor, torch.Tensor]],
    context_mask: torch.Tensor | None,
    video_cache: list[tuple[torch.Tensor, torch.Tensor]],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run the action blocks against the cached video keys and values.

    `tokens` [B, Sa, D], `freqs` [Sa, 1, 2 * head_dim], `context_kv` the text keys and values
    of each block with their [B, 1, Sa, L] bool mask, `mask` the [Sa, Sv + Sa] rows of the
    joint mask.
    """
    x = tokens
    layers = zip(action.blocks, context_kv, video_cache, strict=True)
    for block, kv, (video_k, video_v) in layers:
        q, k, v, state = block.attention_inputs(x, t_mod, freqs)
        k = torch.cat([video_k, k], dim=1)
        v = torch.cat([video_v, v], dim=1)
        x = block.finish(x, attention(q, k, v, action.num_heads, mask), state, kv, context_mask)
    return x
