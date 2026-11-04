# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""The world action model: video and action transformers that predict an action chunk."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from einops import rearrange

from .action_dit import ActionTransformer
from .config import VIDEO_TEXT_LEN, ModelConfig
from .memory import PastFrameMemory
from .mot import joint_attention_mask, project_context, run_action_blocks, run_video_blocks
from .sampler import euler_step, sigma_schedule
from .vae import SPATIAL_COMPRESSION, VideoEncoder
from .wan_dit import VideoTransformer


def tensor_version(tensor: torch.Tensor) -> int | None:
    """The in-place version counter of `tensor`. Inference tensors keep none."""
    return None if tensor.is_inference() else tensor._version


@dataclass
class PromptCache:
    """What a chunk derives from the text alone, for the `text` and `text_mask` it came from.

    `video_kv` is [video blocks, 2, B, VIDEO_TEXT_LEN + 1, H * D]: the cross-attention keys
    and values of each full video block, and `video_pairs` its (keys, values) views per
    block. The last row is the robot state row, which each chunk writes. `context_mask` is the
    [B, 1, chunk, L + 1] mask of the action cross-attention, None when it is all True.
    """

    source: tuple[torch.Tensor, torch.Tensor]
    versions: tuple[int | None, int | None]
    text: torch.Tensor
    context_mask: torch.Tensor | None
    video_kv: torch.Tensor
    video_pairs: list[tuple[torch.Tensor, torch.Tensor]]

    def matches(self, text: torch.Tensor, text_mask: torch.Tensor) -> bool:
        return (
            self.source[0] is text
            and self.source[1] is text_mask
            and self.versions == (tensor_version(text), tensor_version(text_mask))
        )


class GridConstants(NamedTuple):
    """The video pass inputs of one batch size and latent grid that no chunk changes.

    `freqs` [Sv, 1, 2 * head_dim], `t_mod` [B, Sv, 6, D] at flow time 0, `mask` the [Sv, Sv]
    video rows of the joint mask and `action_mask` its [T, Sv + T] action rows.
    """

    freqs: torch.Tensor
    t_mod: torch.Tensor
    mask: torch.Tensor
    action_mask: torch.Tensor


class StepConstants(NamedTuple):
    """The action step inputs of one batch size, prefix length and step count.

    `freqs` [T, 1, 2 * head_dim], one [B, T, 6, D] `t_mods` entry and one sigma `deltas`
    entry per step, and `hold` [B, T, A] True on the progress entries of the prefix rows, or
    None.
    """

    freqs: torch.Tensor
    t_mods: list[torch.Tensor]
    deltas: tuple[torch.Tensor, ...]
    hold: torch.Tensor | None


class WorldActionModel(nn.Module):
    """Predicts a chunk of normalized actions from camera latents, text and robot state.

    The video transformer reads the past-frame memory and the present frame once per chunk.
    The action transformer then denoises the chunk with flow matching, attending to the cached
    video keys and values at every step. No video is generated.

    The model keeps what it derives from the text alone and reuses it for as long as the same
    `text` and `text_mask` tensors come back unchanged. An inference tensor has no version
    counter: give a new tensor for a new prompt. It also keeps the constants of each latent
    grid and step schedule. `to` and `load_state_dict` drop all of them.

    The state dict holds the trained weights only. The VAE encoder is frozen Wan2.2 weights:
    load the trained weights first, then call `load_vae`, which `encode_frames` needs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.video = VideoTransformer(config)
        self.action = ActionTransformer(config)
        self.memory = PastFrameMemory(config)
        self.proprio_proj = nn.Linear(config.proprio_dim, config.text_dim)
        self.register_module("vae", None)
        self._context = project_context
        self._prefill = run_video_blocks
        self._action_step = run_action_blocks
        self._prompt: PromptCache | None = None
        self._grids: dict[tuple, GridConstants] = {}
        self._steps: dict[tuple, StepConstants] = {}

    @classmethod
    def from_config(cls, config: ModelConfig | dict) -> "WorldActionModel":
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        return cls(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.proprio_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.proprio_proj.weight.device

    def _apply(self, fn, recurse=True):
        self._clear_caches()
        return super()._apply(fn, recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._clear_caches()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _clear_caches(self) -> None:
        self._prompt = None
        self._grids.clear()
        self._steps.clear()

    def enable_compile(self, **kwargs) -> None:
        """Compile the context projection, the video prefill and the action step.

        The first chunk is slow while the kernels compile. Later chunks with the same shapes
        are faster. `kwargs` go to `torch.compile`.
        """
        self._context = torch.compile(project_context, **kwargs)
        self._prefill = torch.compile(run_video_blocks, **kwargs)
        self._action_step = torch.compile(run_action_blocks, **kwargs)

    def load_vae(self, path: str | None = None) -> None:
        """Attach the Wan2.2 VAE encoder on the model's device and dtype.

        `path` is a local `Wan2.2_VAE.pth`; None downloads it.
        """
        self.vae = VideoEncoder.from_pretrained(path).to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Encode each frame alone: [B, 3, T, H, W] in [-1, 1] -> [B, 48, T, H / 16, W / 16].

        H and W must be multiples of 16. The present frame is T = 1; past frames encode the
        same way, so a frame encoded once can move from present to memory.
        """
        if self.vae is None:
            raise RuntimeError("No VAE is attached. Call `load_vae` first.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B, 3, T, H, W], got {tuple(video.shape)}")
        if video.shape[3] % SPATIAL_COMPRESSION or video.shape[4] % SPATIAL_COMPRESSION:
            raise ValueError(
                f"H and W must be multiples of {SPATIAL_COMPRESSION}, got {tuple(video.shape)}"
            )
        return self.vae(video.to(device=self.device, dtype=self.dtype, non_blocking=True))

    @torch.no_grad()
    def infer_action(
        self,
        *,
        latents: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        proprio: torch.Tensor,
        noise: torch.Tensor,
        past_latents: torch.Tensor | None = None,
        past_padded: torch.Tensor | None = None,
        prefix: torch.Tensor | None = None,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """Denoise one action chunk. Returns normalized actions [B, T, A], float32, on the
        model device.

        Args:
            latents: the present frame, [B, C, 1, h, w], from `encode_frames`.
            text: text rows from `TextEncoder`, [B, text_len, text_dim].
            text_mask: their mask, [B, text_len] bool.
            proprio: the normalized robot state, [B, proprio_dim].
            noise: standard normal noise, [B, chunk_size, action_dim]. It sets the sample.
            past_latents: the memory frames, [B, C, num_past_frames, h, w], oldest first. None
                starts the memory cold: the present frame fills every slot as padding.
            past_padded: [B, num_past_frames] bool, True where a memory frame holds no real
                observation. Give it with `past_latents`.
            prefix: normalized actions the robot already runs, [B, d, action_dim] with
                1 <= d <= max_prefix_len, or None. The chunk continues from them: the first d
                positions stay pinned to the prefix at flow time 0.
            num_steps: Euler steps of the flow, one action transformer pass each.
        """
        cfg = self.config
        device, dtype = self.device, self.dtype
        batch = latents.shape[0]
        chunk = cfg.chunk_size
        grid = tuple(latents.shape[3:])
        if latents.ndim != 5 or tuple(latents.shape[1:3]) != (cfg.latent_channels, 1):
            raise ValueError(
                f"`latents` must be [B, {cfg.latent_channels}, 1, h, w], got {tuple(latents.shape)}"
            )
        if tuple(text.shape) != (batch, cfg.text_len, cfg.text_dim):
            raise ValueError(
                f"`text` must be [{batch}, {cfg.text_len}, {cfg.text_dim}], got {tuple(text.shape)}"
            )
        if tuple(text_mask.shape) != (batch, cfg.text_len):
            raise ValueError(
                f"`text_mask` must be [{batch}, {cfg.text_len}], got {tuple(text_mask.shape)}"
            )
        if tuple(proprio.shape) != (batch, cfg.proprio_dim):
            raise ValueError(
                f"`proprio` must be [{batch}, {cfg.proprio_dim}], got {tuple(proprio.shape)}"
            )
        if tuple(noise.shape) != (batch, chunk, cfg.action_dim):
            raise ValueError(
                f"`noise` must be [{batch}, {chunk}, {cfg.action_dim}], got {tuple(noise.shape)}"
            )
        if (past_latents is None) != (past_padded is None):
            raise ValueError("Give `past_latents` and `past_padded` together, or neither.")
        delay = 0
        if prefix is not None:
            delay = int(prefix.shape[1])
            if tuple(prefix.shape) != (batch, delay, cfg.action_dim):
                raise ValueError(
                    f"`prefix` must be [{batch}, d, {cfg.action_dim}], got {tuple(prefix.shape)}"
                )
            if not 1 <= delay <= cfg.max_prefix_len:
                raise ValueError(
                    f"The prefix length must be in [1, {cfg.max_prefix_len}], got {delay}"
                )
            prefix = prefix.to(device=device, dtype=dtype, non_blocking=True)

        actions = noise.to(device=device, dtype=dtype, copy=True, non_blocking=True)
        steps = self._step_constants(batch, delay, num_steps)
        # The progress dimension of the prefix rows keeps its first noise: progress is an
        # output, never an input.
        held_noise = None if steps.hold is None else actions.clone()

        # The text rows, then one robot state row.
        prompt = self._prompt_cache(text, text_mask)
        proprio = proprio.to(device=device, dtype=dtype, non_blocking=True)
        proprio_row = self.proprio_proj(proprio.unsqueeze(1))
        row_kv, action_context_kv = self._context(self.video, self.action, prompt.text, proprio_row)
        prompt.video_kv[:, :, :, VIDEO_TEXT_LEN:] = row_kv

        # Memory slots, then the present frame. Every video token sits at flow time 0.
        latents = latents.to(device=device, dtype=dtype, non_blocking=True)
        self.memory.check_grid(*grid)
        if past_latents is None:
            past_latents = latents.expand(-1, -1, cfg.num_past_frames, -1, -1)
            past_padded = torch.ones((batch, cfg.num_past_frames), dtype=torch.bool, device=device)
        else:
            expected = (batch, cfg.latent_channels, cfg.num_past_frames, *grid)
            if tuple(past_latents.shape) != expected:
                raise ValueError(
                    f"`past_latents` must be {expected}, got {tuple(past_latents.shape)}"
                )
            if tuple(past_padded.shape) != (batch, cfg.num_past_frames):
                shape = tuple(past_padded.shape)
                raise ValueError(
                    f"`past_padded` must be [{batch}, {cfg.num_past_frames}], got {shape}"
                )
            past_latents = past_latents.to(device=device, dtype=dtype, non_blocking=True)
            past_padded = past_padded.to(device=device, non_blocking=True)
        past_tokens = self.memory(past_latents, past_padded, self.video.patch_embedding)
        frame = self.video.patch_embedding(latents)
        constants = self._grid_constants(batch, grid, tuple(frame.shape[2:]))
        frame_tokens = rearrange(frame, "b c f h w -> b (f h w) c").contiguous()
        video_cache = self._prefill(
            self.video,
            torch.cat([past_tokens, frame_tokens], dim=1),
            constants.freqs,
            constants.t_mod,
            prompt.video_pairs,
            constants.mask,
        )

        for t_mod, delta in zip(steps.t_mods, steps.deltas, strict=True):
            if delay:
                actions[:, :delay] = prefix
                if held_noise is not None:
                    actions.copy_(torch.where(steps.hold, held_noise, actions))
            tokens = self._action_step(
                self.action,
                self.action.action_embedding(actions),
                steps.freqs,
                t_mod,
                action_context_kv,
                prompt.context_mask,
                video_cache,
                constants.action_mask,
            )
            actions = euler_step(actions, self.action.head(tokens), delta)
            if delay:
                actions[:, :delay] = prefix
                if held_noise is not None:
                    actions = torch.where(steps.hold, held_noise, actions)
        if held_noise is not None:
            actions[:, :delay] = prefix
        return actions.float()

    def _grid_constants(
        self, batch: int, grid: tuple[int, int], frame_grid: tuple[int, int, int]
    ) -> GridConstants:
        """The constants of an h x w latent grid whose present frame patches to `frame_grid`."""
        key = (batch, grid, frame_grid)
        if key not in self._grids:
            device, dtype = self.device, self.dtype
            past_freqs, slot_sizes = self.memory.rope(*grid, device)
            frame_freqs = self.video.frame_rope(*frame_grid, device)
            lengths = (sum(slot_sizes), frame_freqs.shape[0])
            t_mod = torch.cat(
                [
                    self.video.time_modulation(torch.zeros((batch, n), dtype=dtype, device=device))
                    for n in lengths
                ],
                dim=1,
            )
            mask = joint_attention_mask(
                slot_sizes, lengths[1], self.config.chunk_size, self.memory.slot_bias, dtype, device
            )
            video_len = sum(lengths)
            self._grids[key] = GridConstants(
                torch.cat([past_freqs, frame_freqs], dim=0),
                t_mod,
                mask[:video_len, :video_len].contiguous(),
                mask[video_len:],
            )
        return self._grids[key]

    def _step_constants(self, batch: int, delay: int, num_steps: int) -> StepConstants:
        """The constants of `num_steps` flow steps after a prefix of `delay` rows."""
        key = (batch, delay, num_steps)
        if key not in self._steps:
            cfg, device = self.config, self.device
            chunk = cfg.chunk_size
            timesteps, deltas = sigma_schedule(
                num_steps, cfg.sigma_shift, cfg.num_train_timesteps, device, self.dtype
            )
            t_mods = []
            for timestep in timesteps:
                timestep = timestep.reshape(1)
                if delay:
                    timestep = timestep.view(1, 1).expand(batch, chunk).clone()
                    timestep[:, :delay] = 0.0
                t_mod = self.action.time_modulation(timestep, batch)
                if not delay:
                    # The per-position layout of a prefix chunk, so one compiled step serves both.
                    t_mod = t_mod.unsqueeze(1).expand(-1, chunk, -1, -1).contiguous()
                t_mods.append(t_mod)
            hold = None
            if delay and cfg.progress_output:
                hold = torch.zeros((batch, chunk, cfg.action_dim), dtype=torch.bool, device=device)
                hold[..., -1] = torch.arange(chunk, device=device) < delay
            freqs = self.action.chunk_rope(chunk, device)
            self._steps[key] = StepConstants(freqs, t_mods, deltas.unbind(), hold)
        return self._steps[key]

    def _prompt_cache(self, text: torch.Tensor, text_mask: torch.Tensor) -> PromptCache:
        """The cache of `text`, built when the text differs from the last one.

        The video transformer reads the text zero-padded to VIDEO_TEXT_LEN rows; the padded
        mask is all True, so it uses no mask.
        """
        if self._prompt is not None and self._prompt.matches(text, text_mask):
            return self._prompt
        self._prompt = None
        device, dtype = self.device, self.dtype
        batch, text_len, width = text.shape
        source = (text, text_mask)
        versions = (tensor_version(text), tensor_version(text_mask))
        # SDPA can pick the flash kernel only when there is no mask.
        is_open = bool(text_mask.all())
        text = text.to(device=device, dtype=dtype, non_blocking=True)
        text_mask = text_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        context_mask = None
        if not is_open:
            context_mask = torch.cat(
                [text_mask, torch.ones((batch, 1), dtype=torch.bool, device=device)], dim=1
            )
            context_mask = context_mask.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
            context_mask = context_mask.unsqueeze(1)
        padding = text.new_zeros((batch, VIDEO_TEXT_LEN - text_len, width))
        video_text = self.video.text_embedding(torch.cat([text, padding], dim=1))
        blocks = self.video.blocks[:-1]
        inner = self.config.num_heads * self.config.head_dim
        video_kv = text.new_empty((len(blocks), 2, batch, VIDEO_TEXT_LEN + 1, inner))
        for index, block in enumerate(blocks):
            video_kv[index, :, :, :VIDEO_TEXT_LEN] = torch.stack(
                block.cross_attn.project_kv(video_text)
            )
        pairs = [(kv[0], kv[1]) for kv in video_kv]
        self._prompt = PromptCache(source, versions, text, context_mask, video_kv, pairs)
        return self._prompt
