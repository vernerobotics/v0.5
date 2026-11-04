# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Observation history and past-frame memory.

``ObservationHistory`` keeps the recent observations and resamples them onto the 30 Hz
grid the model trained on: poses and grippers by linear interpolation, images by the
nearest observation.

``MemoryBank`` keeps one latent per 133 ms grid tick for the episode and reads 16 of them
back, 1.07 s apart, as the model's past frames.
"""

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .interface import DATA_STEP_S, Observation, chunk_frame_poses
from .transforms import (
    IDENTITY_POSE,
    PARKED_GRIPPER,
    CameraFrames,
    Step,
    orthonormal_pose,
    pose_vector,
)

# Grid steps the history resamples: a 0.6 s chunk period plus 4 steps of margin.
GRID_STEPS = 22
# The history keeps this many seconds of observations. A longer gap between two
# observations raises.
HISTORY_S = GRID_STEPS * DATA_STEP_S + 0.2

NUM_SLOTS = 16
# The ring ticks every 4 data steps (133 ms). The slots are 8 ticks (1.07 s) apart.
TICK_FRAMES = 4
SLOT_TICKS = 8
TICK_S = float(TICK_FRAMES) * DATA_STEP_S
SLOT_S = float(TICK_FRAMES * SLOT_TICKS) * DATA_STEP_S
RING_DEPTH = NUM_SLOTS * SLOT_TICKS + 8


@dataclass(frozen=True)
class _Entry:
    time: float
    frames: CameraFrames
    poses: np.ndarray
    grippers: np.ndarray
    base: np.ndarray


class ObservationHistory:
    """The recent observations of one episode.

    The first observation sets the arms of the episode (one or two) and whether it has a
    base pose. Every later observation must match it.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._entries: list[_Entry] = []
        self.single_arm: bool | None = None
        self.has_base: bool | None = None

    def check(self, obs: Observation) -> None:
        """Raise if ``obs`` cannot be the next observation of the episode."""
        if not isinstance(obs, Observation):
            raise TypeError(f"obs must be an Observation, got {type(obs).__name__}")
        if self._entries and obs.timestamp <= self._entries[-1].time:
            raise ValueError(
                f"timestamp {obs.timestamp} is not after the previous one "
                f"{self._entries[-1].time}. Give each observation once, in time order."
            )
        if self._entries and obs.timestamp - self._entries[-1].time > HISTORY_S:
            raise ValueError(
                f"{obs.timestamp - self._entries[-1].time:.3f} s since the previous "
                f"observation, more than {HISTORY_S:.2f} s. Timestamps are in seconds. Call "
                "reset() when the robot resumes after a pause."
            )
        if self.single_arm is not None and (obs.left is None) != self.single_arm:
            arms = "one arm" if self.single_arm else "two arms"
            raise ValueError(
                f"this episode started with {arms}. Give the same arms in every observation, "
                "or call reset() to start a new episode."
            )
        if self.has_base is not None and (obs.base_pose is not None) != self.has_base:
            raise ValueError(
                "give the base pose in every observation of an episode or in none, or call "
                "reset() to start a new episode."
            )

    def add(self, obs: Observation) -> None:
        """Store one observation. Raises, and stores nothing, if ``check`` fails."""
        self.check(obs)
        left, right, base = chunk_frame_poses(obs)
        if left is None:
            poses = np.stack([IDENTITY_POSE, pose_vector(right)])
            grippers = np.array([PARKED_GRIPPER, obs.right.gripper], dtype=np.float32)
        else:
            poses = np.stack([pose_vector(left), pose_vector(right)])
            grippers = np.array([obs.left.gripper, obs.right.gripper], dtype=np.float32)
        entry = _Entry(obs.timestamp, CameraFrames(obs.images), poses, grippers, pose_vector(base))
        recent = [e for e in self._entries if e.time >= obs.timestamp - HISTORY_S]
        self._entries = recent + [entry]
        self.single_arm = obs.left is None
        self.has_base = obs.base_pose is not None

    def resample(self) -> list[Step]:
        """The ``GRID_STEPS`` grid steps that end at the newest observation, oldest first."""
        if not self._entries:
            raise ValueError("the history is empty: add an observation first")
        times = np.array([e.time for e in self._entries])
        grid = times[-1] - DATA_STEP_S * np.arange(GRID_STEPS - 1, -1, -1)
        nearest = _nearest_indices(times, grid)
        poses = _interp(times, np.stack([e.poses.reshape(-1) for e in self._entries]), grid)
        poses = poses.reshape(GRID_STEPS, 2, -1)
        grippers = _interp(times, np.stack([e.grippers for e in self._entries]), grid)
        base = _interp(times, np.stack([e.base for e in self._entries]), grid)
        return [
            Step(
                time=float(t),
                frames=self._entries[int(nearest[i])].frames,
                poses=np.stack([orthonormal_pose(pose) for pose in poses[i]]),
                grippers=grippers[i],
                base=orthonormal_pose(base[i]),
            )
            for i, t in enumerate(grid)
        ]


def _interp(src_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Linear interpolation per column, clamped at both ends, in the dtype of ``values``."""
    if values.ndim == 1:
        return np.interp(target_t, src_t, values).astype(values.dtype)
    columns = [np.interp(target_t, src_t, values[:, d]) for d in range(values.shape[1])]
    return np.column_stack(columns).astype(values.dtype)


def _nearest_indices(src_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """The index of the nearest source time per target time. A tie takes the later one."""
    idx = np.clip(np.searchsorted(src_t, target_t), 0, len(src_t) - 1)
    prev = np.maximum(idx - 1, 0)
    return np.where(np.abs(src_t[prev] - target_t) < np.abs(src_t[idx] - target_t), prev, idx)


class MemoryBank:
    """Past-frame latents of one episode.

    The grid starts at the first ``read`` of the episode and ticks every ``TICK_S``. Each
    ``read`` adds the grid steps that land on a tick since the previous read, then picks
    for slot ``k`` (1..16) the entry nearest ``now - k * SLOT_S``.

    A slot is padded when its time is before the episode start (it then reads the oldest
    entry) or when its nearest entry is more than half a slot away. When the newest entry
    is more than 1.5 slots old, the bank starts over. With no entry, every slot reads the
    present latent and is padded.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the episode. Call at every episode start."""
        self._entries: deque[tuple[float, torch.Tensor]] = deque(maxlen=RING_DEPTH)
        self._start: float | None = None
        self._tick = 0

    def read(
        self,
        steps: Sequence[Step],
        present_latent: torch.Tensor,
        encode: Callable[[list[Step]], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The past frames for the chunk at ``steps[-1].time``.

        The bank changes only when the read succeeds.

        Args:
            steps: ``ObservationHistory.resample()``, the present step last.
            present_latent: the latent of the present step, ``[1, C, 1, h, w]``.
            encode: grid steps -> their latents, ``[N, C, 1, h, w]``.

        Returns:
            ``past_latents [1, C, 16, h, w]`` and ``past_padded`` bool ``[1, 16]``, the
            oldest slot first.
        """
        now = float(steps[-1].time)
        start, tick, ticks, holds_present = now, 0, [], True
        if self._start is not None:
            start = self._start
            tick, ticks, holds_present = self._grid_ticks(now, steps)
        entries = list(self._entries) + [(step.time, step) for step in ticks]
        entries = entries[-RING_DEPTH:]
        restart = bool(entries) and now - entries[-1][0] > 1.5 * SLOT_S
        if restart:
            start, tick, ticks, entries, holds_present = now, 0, [], [], True
        if ticks:
            latents = encode(ticks)
            encoded = {id(step): latents[i : i + 1] for i, step in enumerate(ticks)}
            entries = [(t, encoded[id(v)] if isinstance(v, Step) else v) for t, v in entries]
        if not entries:
            entries = [(now, present_latent)]

        picked, padded = [], []
        for slot in range(1, NUM_SLOTS + 1):
            target = now - slot * SLOT_S
            index = 0
            if target < start - 0.5 * DATA_STEP_S:
                padded.append(True)
            else:
                for candidate in range(1, len(entries)):
                    if abs(entries[candidate][0] - target) < abs(entries[index][0] - target):
                        index = candidate
                padded.append(abs(entries[index][0] - target) > 0.5 * SLOT_S)
            picked.append(entries[index][1])
        past = torch.cat(picked[::-1], dim=2)
        is_pad = torch.tensor([padded[::-1]], dtype=torch.bool)

        if restart:
            self._entries.clear()
        self._start, self._tick = start, tick
        if ticks:
            self._entries.extend((step.time, encoded[id(step)]) for step in ticks)
        if holds_present:
            self._entries.append((now, present_latent))
        return past, is_pad

    def _grid_ticks(self, now: float, steps: Sequence[Step]) -> tuple[int, list[Step], bool]:
        """The ticks since the last read: the index of the last one, the grid steps on
        them, and whether ``now`` is a tick.

        Tick ``n`` is at ``start + n * TICK_S``. A tick is covered when a grid step is
        within half a tick of it. Only the ticks in reach of ``steps`` can be covered.
        """
        last = math.floor((now + 0.5 * DATA_STEP_S - self._start) / TICK_S)
        reach = math.ceil((steps[0].time - 0.5 * TICK_S - self._start) / TICK_S) - 1
        ticks = []
        for index in range(max(self._tick + 1, reach), last + 1):
            tick = self._start + index * TICK_S
            step = min(steps, key=lambda s: abs(s.time - tick))
            covered = abs(step.time - tick) <= 0.5 * TICK_S
            if covered and step.time >= now:
                return index, ticks, True
            if covered:
                ticks.append(step)
        return max(last, self._tick), ticks, False
