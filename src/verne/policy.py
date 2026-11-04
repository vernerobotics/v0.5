# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Run Verne V0.5 on a robot or in a simulator: observations in, action chunks out."""

import json
import os
import threading
from collections.abc import Iterable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .interface import DATA_STEP_S, ActionChunk, Observation
from .memory_bank import NUM_SLOTS, SLOT_TICKS, MemoryBank, ObservationHistory
from .model import ModelConfig, TextEncoder, WorldActionModel
from .transforms import (
    ACTION_DIM,
    CHUNK_SIZE,
    PROPRIO_DIM,
    Normalizer,
    Step,
    build_canvas,
    decode_actions,
    encode_prefix,
    prefix_rows,
    state_inputs,
)

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
NORMALIZER_PREFIX = "normalizer."
# Free GPU memory the text encoder needs beyond its weights.
TEXT_ENCODER_MARGIN_BYTES = 2 << 30
# Largest error of a prefix row time from obs.timestamp + (j + 1) / 30.
PREFIX_TIME_ATOL_S = 1e-3


class Policy:
    """Verne V0.5 with its text encoder, normalizer, observation history and memory.

    One `Policy` runs one episode at a time. Call `reset` at the start of each episode,
    `observe` for every new frame, and `infer` when you want a new action chunk.

    A `Policy` is safe to share between threads. `observe` holds the history lock only to
    store the frame, so a camera thread can call it while a planner thread runs `infer`.
    `infer`, `reset` and `encode_instructions` run one at a time. Timestamps must still
    rise across the threads.
    """

    def __init__(
        self,
        model: WorldActionModel,
        text_encoder: TextEncoder,
        normalizer: Normalizer,
        prompt_template: str,
        num_steps: int,
        text_encoder_on_gpu: bool = False,
    ) -> None:
        check_config(model.config)
        self.model = model
        self.text_encoder = text_encoder
        self.normalizer = normalizer
        self.prompt_template = prompt_template
        self.num_steps = int(num_steps)
        self.text_encoder_on_gpu = bool(text_encoder_on_gpu)
        self.history = ObservationHistory()
        self.memory = MemoryBank()
        self.generator = torch.Generator()
        self._prompts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        device: str | torch.device = "cuda",
        compile: bool = False,
        text_encoder_on_gpu: bool = False,
        revision: str | None = None,
    ) -> "Policy":
        """Load a checkpoint from a local folder or a Hugging Face repo.

        Args:
            name_or_path: a folder with `config.json` and `model.safetensors`, or a repo id
                such as `vernerobotics/v0.5-base`.
            device: where the model runs. bf16 weights need about 17 GB of GPU memory.
            compile: compile the video prefill and the action step with `torch.compile`.
            text_encoder_on_gpu: keep the text encoder (about 11 GB) on the GPU. When False
                it waits on the CPU and moves to the GPU only to encode a new instruction.
                Each instruction is encoded once.
            revision: the repo revision to download.
        """
        folder = name_or_path
        if not os.path.isdir(folder):
            folder = snapshot_download(
                name_or_path, revision=revision, allow_patterns=[CONFIG_FILE, WEIGHTS_FILE]
            )
        with open(os.path.join(folder, CONFIG_FILE)) as f:
            config = json.load(f)

        # A CPU copy of the weights would hold 17 GB of RAM while the text encoder loads.
        state_dict = load_file(os.path.join(folder, WEIGHTS_FILE), device=str(torch.device(device)))
        normalizer = Normalizer.from_state_dict(state_dict, prefix=NORMALIZER_PREFIX)
        weights = {k: v for k, v in state_dict.items() if not k.startswith(NORMALIZER_PREFIX)}
        del state_dict
        with torch.device("meta"):
            model = WorldActionModel.from_config(config["model"])
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        model = model.to(device).eval().requires_grad_(False)
        model.load_vae()
        if compile:
            model.enable_compile()

        text_device = device if text_encoder_on_gpu else "cpu"
        text_encoder = TextEncoder.from_pretrained(model.config.text_len, text_device)
        return cls(
            model,
            text_encoder,
            normalizer,
            prompt_template=config["prompt_template"],
            num_steps=config["num_steps"],
            text_encoder_on_gpu=text_encoder_on_gpu,
        )

    def reset(self, seed: int = 0) -> None:
        """Start a new episode: forget the observation history and the memory.

        Also call it when the robot resumes after a pause. `seed` sets the sampling noise.
        """
        with self._lock, self._history_lock:
            self.history.reset()
            self.memory.reset()
            self.generator.manual_seed(int(seed))

    def observe(self, obs: Observation) -> None:
        """Add one observation without planning. Call it for every frame between `infer`s."""
        with self._history_lock:
            self.history.add(obs)

    def encode_instructions(self, instructions: Iterable[str]) -> None:
        """Encode instructions before the episode, so that `infer` does not wait for the
        text encoder. `infer` encodes a new instruction itself, once."""
        instructions = list(instructions)
        bad = [text for text in instructions if not isinstance(text, str) or not text.strip()]
        if bad:
            raise ValueError(f"an instruction must be a non-empty string, got {bad}")
        prompts = [self.prompt_template.format(instruction=text) for text in instructions]
        with self._lock:
            self._encode_prompts([p for p in dict.fromkeys(prompts) if p not in self._prompts])

    @torch.no_grad()
    def infer(self, obs: Observation, prefix: ActionChunk | None = None) -> ActionChunk:
        """Add `obs` and plan a new chunk of 32 actions at 30 Hz.

        Every input is checked before the policy changes, so a call that raises leaves the
        policy as it was.

        Args:
            obs: the observation now.
            prefix: the first rows of the new chunk that the robot runs no matter what,
                usually the next rows of the chunk it runs now:
                ``chunk.at(obs.timestamp + np.arange(1, d + 1) / 30)``, with
                ``1 <= d <= max_prefix_len``. The model keeps them and plans the rest after
                them. None plans the whole chunk.
        """
        with self._lock:
            return self._infer(obs, prefix)

    def _infer(self, obs: Observation, prefix: ActionChunk | None) -> ActionChunk:
        with self._history_lock:
            self.history.check(obs)
        if prefix is not None:
            self._check_prefix(obs, prefix)
        text, text_mask = self._text(obs.instruction)
        with self._history_lock:
            self.history.add(obs)
            steps = self.history.resample()
        present = self._encode(steps[-1:])
        past, past_padded = self.memory.read(steps, present, self._encode)
        single_arm = obs.left is None
        proprio, reference = state_inputs(
            steps[-2], steps[-1], self.normalizer, single_arm=single_arm
        )
        base_reference = steps[-1].base.astype(np.float32)
        model_prefix, relative_prefix = None, None
        if prefix is not None:
            rows, base, progress = prefix_rows(prefix)
            if obs.base_pose is None:
                base = None
            model_prefix, *relative_prefix = encode_prefix(
                rows, base, progress, reference, base_reference, self.normalizer
            )
        config = self.model.config
        noise = torch.randn((1, config.chunk_size, config.action_dim), generator=self.generator)
        actions = self.model.infer_action(
            latents=present,
            text=text,
            text_mask=text_mask,
            proprio=proprio,
            noise=noise,
            past_latents=past,
            past_padded=past_padded,
            prefix=model_prefix,
            num_steps=self.num_steps,
        )
        return decode_actions(
            actions,
            reference,
            self.normalizer,
            obs.timestamp,
            base_reference=base_reference,
            prefix=relative_prefix,
            single_arm=single_arm,
        )

    def _check_prefix(self, obs: Observation, prefix: ActionChunk) -> None:
        if not isinstance(prefix, ActionChunk):
            raise TypeError(f"prefix must be an ActionChunk, got {type(prefix).__name__}")
        limit = self.model.config.max_prefix_len
        if not 1 <= len(prefix) <= limit:
            raise ValueError(
                f"the prefix holds {len(prefix)} rows; this checkpoint takes 1 to {limit}"
            )
        expected = obs.timestamp + DATA_STEP_S * np.arange(1, len(prefix) + 1)
        error = float(np.abs(prefix.timestamps - expected).max())
        if error > PREFIX_TIME_ATOL_S:
            raise ValueError(
                f"prefix row j must be at obs.timestamp + (j + 1) / 30, but a row is "
                f"{error * 1e3:.1f} ms off. Take it with "
                "chunk.at(obs.timestamp + np.arange(1, d + 1) / 30)."
            )
        if (prefix.left is None) != (obs.left is None):
            raise ValueError("the prefix must hold the same arms as the observation")

    def _encode(self, steps: list[Step]) -> torch.Tensor:
        video = torch.stack([build_canvas(step.frames.prepared()) for step in steps])[:, :, None]
        return self.model.encode_frames(video)

    def _text(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The encoded prompt for `instruction`. Each prompt is encoded once and kept."""
        prompt = self.prompt_template.format(instruction=instruction)
        if prompt not in self._prompts:
            self._encode_prompts([prompt])
        return self._prompts[prompt]

    def _encode_prompts(self, prompts: list[str]) -> None:
        """Encode and keep the prompts, on the model's GPU. When the GPU cannot hold the
        model and the text encoder, the model waits on the CPU while the encoder runs."""
        if not prompts:
            return
        device = self.model.device
        if self.text_encoder_on_gpu or device.type != "cuda":
            encoded = [self.text_encoder(prompt) for prompt in prompts]
        else:
            need = sum(p.numel() * p.element_size() for p in self.text_encoder.parameters())
            swap = torch.cuda.mem_get_info(device)[0] < need + TEXT_ENCODER_MARGIN_BYTES
            if swap:
                self.model.to("cpu")
            try:
                self.text_encoder.to(device)
                try:
                    encoded = [self.text_encoder(prompt) for prompt in prompts]
                finally:
                    self.text_encoder.to("cpu")
                    torch.cuda.empty_cache()
            finally:
                if swap:
                    self.model.to(device)
        for prompt, (text, mask) in zip(prompts, encoded, strict=True):
            self._prompts[prompt] = (text.to(device), mask.to(device))


def check_config(config: ModelConfig) -> None:
    """Raise if the checkpoint config differs from the constants of this package."""
    expected = {
        "num_past_frames": NUM_SLOTS,
        "past_frame_stride": SLOT_TICKS,
        "chunk_size": CHUNK_SIZE,
        "action_dim": ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
        "progress_output": True,
    }
    wrong = {
        name: getattr(config, name)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if wrong:
        raise ValueError(
            f"the checkpoint config {wrong} differs from this package {expected}. Load a V0.5 "
            "checkpoint, or the verne package version that the checkpoint was made for."
        )
