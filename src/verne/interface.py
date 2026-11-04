# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""The observation and action chunk types of Verne V0.5.

`Policy.observe` and `Policy.infer` read an `Observation`. `Policy.infer` returns an
`ActionChunk` of 32 rows at 30 Hz. Each type checks its values when it is made. See
docs/interface.md for the frames, the units and real-time action chunking.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .geometry import floor_pose, matrix_to_pose, pose_to_matrix

DATA_RATE_HZ = 30.0
DATA_STEP_S = 1.0 / DATA_RATE_HZ
CAMERA_KEYS = ("camera_0", "camera_1", "camera_2")
# Largest error of a pose's bottom row and of R^T R from the identity.
RIGID_ATOL = 1e-3


@dataclass(frozen=True)
class ArmState:
    """The measured state of one arm.

    Attributes:
        pose: the tool pose, a float64 ``[4, 4]`` homogeneous transform in meters, in the
            robot base frame (+x forward, +y left, +z up). Tool axes: +z points out of the
            gripper along the approach direction. When the gripper points forward and
            level, +x points down and +y points left.
        gripper: the jaw opening, from 0.0 (closed) to 1.0 (fully open).
    """

    pose: np.ndarray
    gripper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _rigid(self.pose, "ArmState.pose", batched=False))
        gripper = np.asarray(self.gripper, dtype=np.float64)
        if gripper.size != 1 or not np.isfinite(gripper).all():
            raise ValueError(f"ArmState.gripper must be one finite number, got {self.gripper}")
        object.__setattr__(self, "gripper", float(gripper.reshape(-1)[0]))


@dataclass(frozen=True)
class Observation:
    """What the robot sees and measures at one instant.

    Attributes:
        timestamp: the capture time in seconds, on a clock that only increases.
        instruction: the task in plain English.
        images: uint8 RGB ``[H, W, 3]`` images of any size, by camera: ``camera_0`` on the
            left wrist, ``camera_1`` on the right wrist, ``camera_2`` the head camera, a
            fixed view of the whole work area. Each image is stretched to 256 x 256, so
            give the full field of view. A missing camera, or None, is a black image. The
            images are copied.
        right: the right arm. A single-arm robot gives its arm here.
        left: the left arm, or None for a single-arm robot. A single-arm robot gives no
            ``camera_0``. The model then reads a parked left arm at the identity pose with
            an open jaw and a black left wrist camera.
        base_pose: a mobile robot only: the float64 ``[4, 4]`` pose of the robot base frame
            in a fixed frame such as odometry. V0.5 reads its x, y and yaw, the pose on the
            floor. Each arm pose stays in the robot base frame. `Policy` puts the tools in
            the fixed frame (floor base pose @ arm pose), and the action chunk comes back
            in the fixed frame.

    Give the same arms, and a base pose or none, in every observation of an episode.
    """

    timestamp: float
    instruction: str
    images: Mapping[str, np.ndarray | None]
    right: ArmState
    left: ArmState | None = None
    base_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError(f"Observation.timestamp must be finite, got {self.timestamp}")
        object.__setattr__(self, "timestamp", timestamp)
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError(
                f"Observation.instruction must be a non-empty string, got {self.instruction!r}"
            )
        _require(self.right, ArmState, "Observation.right")
        if self.left is not None:
            _require(self.left, ArmState, "Observation.left")
        if self.base_pose is not None:
            pose = _rigid(self.base_pose, "Observation.base_pose", batched=False)
            object.__setattr__(self, "base_pose", pose)
        object.__setattr__(self, "images", _images(self.images, single_arm=self.left is None))


@dataclass(frozen=True)
class ArmTargets:
    """The targets of one arm, one per chunk row.

    Attributes:
        pose: float64 ``[T, 4, 4]`` tool poses with the tool axes of `ArmState.pose`.
            With `Observation.base_pose` they are in its fixed frame: to command an arm in
            the robot base frame, use ``inv(floor_pose(measured base pose)) @ pose``, with
            `verne.geometry.floor_pose`. Without it they are in the robot base frame.
        gripper: float32 ``[T]`` jaw openings, from 0.0 (closed) to 1.0 (fully open).
    """

    pose: np.ndarray
    gripper: np.ndarray

    def __post_init__(self) -> None:
        pose = _rigid(self.pose, "ArmTargets.pose", batched=True)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "gripper", _series(self.gripper, len(pose), "ArmTargets.gripper"))

    def __len__(self) -> int:
        return len(self.pose)


@dataclass(frozen=True)
class ActionChunk:
    """Targets on a time grid. `Policy.infer` returns 32 rows at 30 Hz: row ``j`` is the
    target for ``obs.timestamp + (j + 1) / 30``.

    Attributes:
        timestamps: float64 ``[T]``, the target time of each row in seconds, rising.
        right: the right arm targets.
        left: the left arm targets, None for a single-arm robot.
        base_pose: float64 ``[T, 4, 4]`` base targets on the floor (x, y and yaw). With
            `Observation.base_pose` they are in its fixed frame. Without it they are
            relative to the base pose at ``obs.timestamp``.
        progress: float32 ``[T]``, how much of the current instruction is done, 0 to 1.
    """

    timestamps: np.ndarray
    right: ArmTargets
    left: ArmTargets | None
    base_pose: np.ndarray
    progress: np.ndarray

    def __post_init__(self) -> None:
        times = np.array(self.timestamps, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError(f"ActionChunk.timestamps must be finite [T], got {times.shape}")
        if np.any(np.diff(times) <= 0):
            raise ValueError("ActionChunk.timestamps must rise")
        times.flags.writeable = False
        object.__setattr__(self, "timestamps", times)
        _require(self.right, ArmTargets, "ActionChunk.right")
        if self.left is not None:
            _require(self.left, ArmTargets, "ActionChunk.left")
        base = _rigid(self.base_pose, "ActionChunk.base_pose", batched=True)
        object.__setattr__(self, "base_pose", base)
        for name in ("right", "left", "base_pose"):
            rows = getattr(self, name)
            if rows is not None and len(rows) != len(times):
                raise ValueError(
                    f"ActionChunk.{name} holds {len(rows)} rows, the timestamps {len(times)}"
                )
        progress = _series(self.progress, len(times), "ActionChunk.progress")
        object.__setattr__(self, "progress", progress)

    def __len__(self) -> int:
        return len(self.timestamps)

    def at(self, times) -> "ActionChunk":
        """The chunk at other ``times``, rising.

        Poses, grippers and progress interpolate linearly between rows. Before the first
        row and after the last one they hold that row. Use it to take the prefix for
        `Policy.infer` at ``obs.timestamp + (j + 1) / 30``, or the target of one control
        tick.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        rows_t = self.timestamps

        def poses(mats: np.ndarray) -> np.ndarray:
            vectors = matrix_to_pose(mats)
            return pose_to_matrix(
                np.column_stack([np.interp(times, rows_t, vectors[:, k]) for k in range(9)])
            )

        def values(series: np.ndarray) -> np.ndarray:
            return np.interp(times, rows_t, series.astype(np.float64)).astype(np.float32)

        def arm(targets: ArmTargets | None) -> ArmTargets | None:
            if targets is None:
                return None
            return ArmTargets(poses(targets.pose), values(targets.gripper))

        return ActionChunk(
            times, arm(self.right), arm(self.left), poses(self.base_pose), values(self.progress)
        )

    @classmethod
    def hold(cls, obs: Observation, num_rows: int) -> "ActionChunk":
        """The robot held still at ``obs``: its poses and grippers, ``num_rows`` times.

        Rows are at ``obs.timestamp + (j + 1) / 30``, in the frame `Policy.infer` returns.
        Progress is 0. Give it as the prefix of the first chunk of an episode, while the
        robot holds still during planning.
        """
        _require(obs, Observation, "obs")
        num_rows = int(num_rows)
        if num_rows < 1:
            raise ValueError(f"num_rows must be 1 or more, got {num_rows}")
        left, right, base = chunk_frame_poses(obs)
        times = obs.timestamp + DATA_STEP_S * np.arange(1, num_rows + 1)

        def repeat(pose: np.ndarray) -> np.ndarray:
            return np.repeat(pose[None], num_rows, axis=0)

        def arm(pose: np.ndarray, state: ArmState) -> ArmTargets:
            return ArmTargets(repeat(pose), np.full(num_rows, state.gripper, dtype=np.float32))

        return cls(
            times,
            arm(right, obs.right),
            None if obs.left is None else arm(left, obs.left),
            repeat(base),
            np.zeros(num_rows, dtype=np.float32),
        )


def chunk_frame_poses(obs: Observation) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """The left tool, right tool and base poses of ``obs`` in the frame `Policy.infer`
    returns, float64 ``[4, 4]`` each.

    With a base pose this is the fixed frame: the base is projected to the floor, and each
    tool pose is ``floor base pose @ arm pose``, as in training. Without one it is the
    robot base frame, and the base is the identity. ``left`` is None for a single-arm
    robot.
    """
    base = np.eye(4) if obs.base_pose is None else floor_pose(obs.base_pose)
    left = None if obs.left is None else base @ obs.left.pose
    return left, base @ obs.right.pose, base


def _require(value, cls: type, name: str) -> None:
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be a {cls.__name__}, got {type(value).__name__}")


def _rigid(value, name: str, batched: bool) -> np.ndarray:
    """``value`` as read-only float64 rigid transforms, ``[T, 4, 4]`` or ``[4, 4]``."""
    pose = np.array(value, dtype=np.float64)
    if pose.shape[-2:] != (4, 4) or pose.ndim != (3 if batched else 2) or not pose.size:
        shape = "[T, 4, 4]" if batched else "[4, 4]"
        raise ValueError(f"{name} must be a {shape} homogeneous transform, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} holds a NaN or an infinite value")
    flat = pose.reshape(-1, 4, 4)
    bottom = np.abs(flat[:, 3] - [0.0, 0.0, 0.0, 1.0]).max(axis=-1)
    if bottom.max() > RIGID_ATOL:
        row = flat[int(np.argmax(bottom)), 3].round(4).tolist()
        raise ValueError(
            f"the bottom row of {name} must be [0, 0, 0, 1], got {row}. Put the position in "
            "the last column: a position in the bottom row means the matrix is transposed."
        )
    rot = flat[:, :3, :3]
    det = np.linalg.det(rot)
    if det.min() <= 0:
        raise ValueError(
            f"the rotation of {name} has determinant {det.min():.3f}: one axis is flipped. "
            "Give a right-handed frame with the tool axes of ArmState."
        )
    error = np.abs(np.swapaxes(rot, -1, -2) @ rot - np.eye(3)).max()
    if error > RIGID_ATOL:
        raise ValueError(
            f"the rotation of {name} is not orthonormal: R^T R differs from the identity by "
            f"{error:.2g}. Give a rotation matrix, without scale or shear."
        )
    pose.flags.writeable = False
    return pose


def _series(value, length: int, name: str) -> np.ndarray:
    """``value`` as a read-only finite float32 ``[length]`` array."""
    series = np.array(value, dtype=np.float32)
    if series.shape != (length,) or not np.isfinite(series).all():
        raise ValueError(f"{name} must be finite [{length}], got shape {series.shape}")
    series.flags.writeable = False
    return series


def _images(images: Mapping, single_arm: bool) -> dict[str, np.ndarray | None]:
    """Read-only copies of the camera images, one per ``CAMERA_KEYS`` entry, None when a
    camera is missing."""
    if not isinstance(images, Mapping):
        raise TypeError(f"Observation.images must map camera keys to images, got {type(images)}")
    unknown = sorted(set(images) - set(CAMERA_KEYS))
    if unknown:
        raise KeyError(f"unknown camera keys {unknown}; expected {list(CAMERA_KEYS)}")
    out = {}
    for key in CAMERA_KEYS:
        image = images.get(key)
        if image is not None:
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or not image.size:
                raise ValueError(
                    f"{key} must be a uint8 [H, W, 3] RGB image, got {image.dtype} {image.shape}"
                )
            image = image.copy()
            image.flags.writeable = False
        out[key] = image
    if single_arm and out["camera_0"] is not None:
        raise ValueError(
            "camera_0 is the left wrist camera, and this observation has no left arm. Drop "
            "camera_0, or give the left arm."
        )
    return out
