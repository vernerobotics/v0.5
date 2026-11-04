# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""The model configuration, stored as `config.json` next to the weights."""

from dataclasses import asdict, dataclass, fields
from typing import Any

# The frozen VAE and the text encoder come from this Hugging Face repo, at this revision.
BACKBONE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
BACKBONE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"

# Memory pack kernels, as multiples of the video patch size on each axis.
PACK_KERNELS = (1, 2, 4)
# The video transformer reads the text zero-padded to this many tokens.
VIDEO_TEXT_LEN = 512


@dataclass(frozen=True)
class ModelConfig:
    """The shape and sampling settings of one checkpoint. Every field is required."""

    # Attention geometry. Both transformers use the same heads.
    num_heads: int
    head_dim: int
    eps: float
    # Conditioning: text rows from the text encoder, one robot state row, the time embedding.
    text_dim: int
    text_len: int
    proprio_dim: int
    freq_dim: int
    # Video transformer.
    latent_channels: int
    patch_size: tuple[int, int, int]
    video_dim: int
    video_ffn_dim: int
    # The video block that each action layer attends to, one entry per action layer.
    paired_video_blocks: tuple[int, ...]
    # Action transformer.
    action_dim: int
    chunk_size: int
    action_hidden_dim: int
    action_ffn_dim: int
    # Past-frame memory: `num_past_frames` frames, `past_frame_stride` video frames apart.
    # Each schedule entry is (frames, time kernel, space kernel), newest frames first.
    num_past_frames: int
    past_frame_stride: int
    memory_schedule: tuple[tuple[int, int, int], ...]
    # Flow-matching sampler.
    sigma_shift: float
    num_train_timesteps: int
    # The longest action prefix the model was trained to continue.
    max_prefix_len: int
    # True when the last action dimension is a progress value that the model predicts but
    # never reads as input.
    progress_output: bool

    def __post_init__(self) -> None:
        if self.head_dim % 2:
            raise ValueError(f"`head_dim` must be even for RoPE, got {self.head_dim}")
        blocks = self.paired_video_blocks
        if not blocks or blocks[0] < 0 or any(b <= a for a, b in zip(blocks, blocks[1:])):
            raise ValueError(f"`paired_video_blocks` must be non-negative and rise, got {blocks}")
        if not 0 < self.text_len <= VIDEO_TEXT_LEN:
            raise ValueError(f"`text_len` must be in [1, {VIDEO_TEXT_LEN}], got {self.text_len}")
        if not 0 <= self.max_prefix_len < self.chunk_size:
            raise ValueError(
                f"`max_prefix_len` must be in [0, {self.chunk_size}), got {self.max_prefix_len}"
            )
        if self.num_past_frames <= 0 or self.past_frame_stride <= 0:
            raise ValueError("`num_past_frames` and `past_frame_stride` must be positive")
        total = 0
        for frames, t_kernel, s_kernel in self.memory_schedule:
            if t_kernel not in PACK_KERNELS or s_kernel not in PACK_KERNELS:
                raise ValueError(
                    f"Memory kernels must be in {PACK_KERNELS}, got {t_kernel}, {s_kernel}"
                )
            if frames <= 0 or frames % t_kernel:
                raise ValueError(
                    f"A schedule entry must hold whole kernels, got {frames}, {t_kernel}"
                )
            total += frames
        if total != self.num_past_frames:
            raise ValueError(f"`memory_schedule` covers {total} frames, not {self.num_past_frames}")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        names = {field.name for field in fields(cls)}
        if set(values) != names:
            raise ValueError(
                f"Config keys differ: missing {sorted(names - set(values))}, "
                f"unknown {sorted(set(values) - names)}"
            )
        values = dict(values)
        values["patch_size"] = tuple(int(p) for p in values["patch_size"])
        values["paired_video_blocks"] = tuple(int(b) for b in values["paired_video_blocks"])
        values["memory_schedule"] = tuple(
            tuple(int(v) for v in entry) for entry in values["memory_schedule"]
        )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["patch_size"] = list(self.patch_size)
        values["paired_video_blocks"] = list(self.paired_video_blocks)
        values["memory_schedule"] = [list(entry) for entry in self.memory_schedule]
        return values
