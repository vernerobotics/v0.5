# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Transforms between the `verne.interface` types and the model inputs and outputs."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import (
    floor_pose,
    matrix_to_pose,
    matrix_to_rot6d,
    nearest_rotation,
    pose_to_matrix,
    rot6d_to_matrix,
)
from .interface import CAMERA_KEYS, DATA_STEP_S, ActionChunk, ArmTargets

CHUNK_SIZE = 32

CAMERA_SIZE = (256, 256)
CANVAS_SIZE = (CAMERA_SIZE[0], CAMERA_SIZE[1] * len(CAMERA_KEYS))
# The head camera is centre-cropped to this fraction of each side, then resized back.
CROPPED_CAMERAS = ("camera_2",)
CROP_RATIO = 0.875

ARMS = ("left", "right")
# Model action row: left pose (9), left gripper, right pose (9), right gripper, base pose
# (9), progress. A pose is xyz then the first two rows of the rotation matrix.
POSE_DIM = 9
ARM_ROW_DIM = 2 * (POSE_DIM + 1)
POSE_COLUMNS = (0, POSE_DIM + 1)
GRIPPER_COLUMNS = (POSE_DIM, 2 * POSE_DIM + 1)
BASE_COLUMNS = slice(ARM_ROW_DIM, ARM_ROW_DIM + POSE_DIM)
PROGRESS_COLUMN = ARM_ROW_DIM + POSE_DIM
ACTION_DIM = PROGRESS_COLUMN + 1
# Robot state per step: left pose, left gripper, right pose, right gripper, and the right
# pose in the frame of the left pose. Two steps, the previous one first.
STATE_ROW_DIM = ARM_ROW_DIM + POSE_DIM
PROPRIO_DIM = 2 * STATE_ROW_DIM

IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
PARKED_GRIPPER = 1.0
NORMALIZER_CLAMP = 5.0

# OpenCV INTER_LINEAR fixed-point arithmetic for uint8 images.
_COEF_SCALE = 2048


class CameraFrames:
    """The camera images of one observation. They are resized to 256 x 256 when a canvas
    first reads them, then kept."""

    def __init__(self, images: Mapping[str, np.ndarray | None]) -> None:
        self._images = images
        self._prepared: dict[str, np.ndarray] | None = None

    def prepared(self) -> dict[str, np.ndarray]:
        """The images from ``prepare_images``."""
        if self._prepared is None:
            self._prepared = prepare_images(self._images)
            self._images = None
        return self._prepared


@dataclass(frozen=True)
class Step:
    """One step of the 30 Hz observation grid.

    ``frames``: the camera images of the nearest observation. ``poses``: float64 ``[2, 9]``
    from ``orthonormal_pose``, left arm first. ``grippers``: float32 ``[2]``. ``base``:
    float64 ``[9]`` from ``orthonormal_pose``, the identity pose without a base input.
    """

    time: float
    frames: CameraFrames
    poses: np.ndarray
    grippers: np.ndarray
    base: np.ndarray


def _resize_taps(src: int, dst: int, clamp_weights: bool) -> tuple[np.ndarray, ...]:
    """Source taps and fixed-point weights of OpenCV's bilinear resize along one axis.

    Columns clamp the tap AND zero the weight at the borders. Rows only clamp the tap, so
    an edge row blends a source row with itself.
    """
    pos = ((np.arange(dst) + 0.5) * (1.0 / (dst / src)) - 0.5).astype(np.float32)
    lo = np.floor(pos).astype(np.int64)
    frac = (pos - lo).astype(np.float32)
    if clamp_weights:
        below = lo < 0
        frac[below] = 0.0
        lo[below] = 0
        above = lo >= src - 1
        frac[above] = 0.0
        lo[above] = src - 1
    w0 = np.rint((1.0 - frac) * _COEF_SCALE).astype(np.int32)
    w1 = np.rint(frac * _COEF_SCALE).astype(np.int32)
    return np.clip(lo, 0, src - 1), np.clip(lo + 1, 0, src - 1), w0, w1


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a uint8 ``[H, W, C]`` image with the integer arithmetic of
    ``cv2.resize(..., interpolation=cv2.INTER_LINEAR)``, which built the training images.
    No antialiasing."""
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) == (height, width):
        return image.copy()
    x0, x1, wx0, wx1 = _resize_taps(src_w, width, clamp_weights=True)
    y0, y1, wy0, wy1 = _resize_taps(src_h, height, clamp_weights=False)
    pixels = image.astype(np.int32)
    rows = pixels[:, x0] * wx0[None, :, None] + pixels[:, x1] * wx1[None, :, None]
    top = ((rows[y0] >> 4) * wy0[:, None, None]) >> 16
    bottom = ((rows[y1] >> 4) * wy1[:, None, None]) >> 16
    return np.clip((top + bottom + 2) >> 2, 0, 255).astype(np.uint8)


def prepare_images(images: Mapping[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    """``Observation.images`` -> uint8 256 x 256 RGB per camera. A missing camera is black."""
    out = {}
    for key in CAMERA_KEYS:
        image = images.get(key)
        if image is None:
            out[key] = np.zeros(CAMERA_SIZE + (3,), dtype=np.uint8)
        else:
            out[key] = resize_image(image, *CAMERA_SIZE)
    return out


def build_canvas(images: Mapping[str, np.ndarray]) -> torch.Tensor:
    """Prepared images (``prepare_images``) -> float32 ``[3, 256, 768]`` in [-1, 1].

    The cameras sit side by side in ``CAMERA_KEYS`` order. The head camera is centre
    cropped and resized back to 256 x 256 with antialiased bilinear interpolation.
    """
    height, width = CAMERA_SIZE
    canvas = torch.empty((3,) + CANVAS_SIZE, dtype=torch.float32)
    for i, key in enumerate(CAMERA_KEYS):
        image = np.asarray(images[key])
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 {(height, width, 3)}, got {image.shape}")
        if key in CROPPED_CAMERAS:
            crop_h = max(1, int(round(height * CROP_RATIO)))
            crop_w = max(1, int(round(width * CROP_RATIO)))
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            image = image[top : top + crop_h, left : left + crop_w]
        frame = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
        frame = frame.to(torch.float32).div_(255.0)
        if frame.shape[-2:] != (height, width):
            frame = F.interpolate(
                frame, size=[height, width], mode="bilinear", align_corners=False, antialias=True
            )
        canvas[:, :, i * width : (i + 1) * width] = frame[0]
    return canvas.sub_(0.5).div_(0.5)


def pose_vector(pose: np.ndarray) -> np.ndarray:
    """A ``[4, 4]`` pose -> float32 ``[9]`` pose vector."""
    return np.concatenate([pose[:3, 3], matrix_to_rot6d(pose[:3, :3])]).astype(np.float32)


def orthonormal_pose(pose: np.ndarray) -> np.ndarray:
    """A ``[9]`` pose vector -> float64 ``[9]`` with an orthonormal rotation."""
    rot = nearest_rotation(rot6d_to_matrix(pose[3:9]))
    return np.concatenate([np.asarray(pose[:3], dtype=np.float64), matrix_to_rot6d(rot)])


def inter_arm_pose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The right pose in the frame of the left pose, float32 ``[9]``."""
    poses = []
    for vec in (left, right):
        mat = np.eye(4)
        mat[:3, :3] = rot6d_to_matrix(vec[3:9])
        mat[:3, 3] = vec[:3]
        poses.append(mat)
    rel = np.linalg.inv(poses[0]) @ poses[1]
    rot = matrix_to_rot6d(nearest_rotation(rel[:3, :3]))
    return np.concatenate([rel[:3, 3], rot]).astype(np.float32)


def relative_pose_history(poses: np.ndarray) -> np.ndarray:
    """float32 ``[T, 9]`` poses -> each pose in the frame of the last one, float32."""
    mats = pose_to_matrix(poses)
    rel = np.linalg.inv(mats[-1])[None] @ mats
    return matrix_to_pose(rel).astype(poses.dtype)


def build_proprio(poses: np.ndarray, grippers: np.ndarray, shared_frame: bool) -> np.ndarray:
    """The robot state input, float32 ``[58]``.

    Args:
        poses: float64 ``[2, 2, 9]`` (step, arm) from ``orthonormal_pose``, the previous
            step first and the left arm first.
        grippers: float32 ``[2, 2]`` (step, arm).
        shared_frame: False zeroes the pose between the arms, for a robot whose two arm
            poses are not in one frame, such as a single arm with a parked placeholder.
    """
    if poses.shape != (2, 2, POSE_DIM) or grippers.shape != (2, 2):
        raise ValueError(f"poses must be [2, 2, 9] and grippers [2, 2], got {poses.shape}")
    if shared_frame:
        inter = np.stack([inter_arm_pose(poses[s, 0], poses[s, 1]) for s in range(2)])
    else:
        inter = np.zeros((2, POSE_DIM), dtype=np.float32)
    grippers = grippers.astype(np.float32)
    tracks = [
        relative_pose_history(poses[:, 0].astype(np.float32)),
        grippers[:, 0:1],
        relative_pose_history(poses[:, 1].astype(np.float32)),
        grippers[:, 1:2],
        inter,
    ]
    return np.concatenate(tracks, axis=-1).reshape(-1).astype(np.float32)


def reference_row(poses: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    """The current state as one float32 ``[20]`` arm row: the frame of every relative pose.

    Args:
        poses: float64 ``[2, 9]`` from ``orthonormal_pose``, left arm first.
        grippers: float32 ``[2]``.
    """
    return np.concatenate(
        [
            poses[0].astype(np.float32),
            grippers[0:1].astype(np.float32),
            poses[1].astype(np.float32),
            grippers[1:2].astype(np.float32),
        ]
    )


def state_inputs(
    previous: Step, current: Step, normalizer: "Normalizer", single_arm: bool = False
) -> tuple[torch.Tensor, np.ndarray]:
    """The normalized robot state input ``[1, 58]`` and the reference row ``[20]``."""
    proprio = build_proprio(
        np.stack([previous.poses, current.poses]),
        np.stack([previous.grippers, current.grippers]),
        shared_frame=not single_arm,
    )
    proprio = normalizer.normalize_proprio(torch.from_numpy(proprio)[None])
    return proprio, reference_row(current.poses, current.grippers)


def _compose(rows: np.ndarray, reference: np.ndarray, inverse: bool) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != ARM_ROW_DIM or reference.shape != (ARM_ROW_DIM,):
        raise ValueError(f"rows must be [T, 20] and reference [20], got {rows.shape}")
    out = rows.copy()
    for start in POSE_COLUMNS:
        span = slice(start, start + POSE_DIM)
        anchor = pose_to_matrix(reference[span][None])[0]
        if inverse:
            anchor = np.linalg.inv(anchor)
        out[:, span] = matrix_to_pose(anchor @ pose_to_matrix(rows[:, span]))
    return out


def relative_poses(poses: np.ndarray, anchor: np.ndarray, inverse: bool = True) -> np.ndarray:
    """``[T, 9]`` poses -> ``inv(anchor) @ pose`` per row, or ``anchor @ pose`` when not
    ``inverse``. ``anchor`` is one ``[9]`` pose. float32."""
    anchor = pose_to_matrix(np.asarray(anchor, dtype=np.float32)[None])[0]
    if inverse:
        anchor = np.linalg.inv(anchor)
    poses = np.asarray(poses, dtype=np.float32)
    return matrix_to_pose(anchor @ pose_to_matrix(poses)).astype(np.float32)


def to_relative(rows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Absolute ``[T, 20]`` arm rows -> rows relative to ``reference`` (``reference_row``).
    Each pose becomes ``inv(reference pose) @ pose``. Grippers pass through."""
    return _compose(rows, reference, inverse=True)


def to_absolute(rows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Relative ``[T, 20]`` arm rows -> absolute rows: ``reference pose @ pose``."""
    return _compose(rows, reference, inverse=False)


def to_model_rows(rows: np.ndarray, base: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """``[T, 20]`` arm rows, ``[T, 9]`` base poses and ``[T]`` progress -> float32
    ``[T, 30]`` model rows."""
    out = np.empty((len(rows), ACTION_DIM), dtype=np.float32)
    out[:, :ARM_ROW_DIM] = rows
    out[:, BASE_COLUMNS] = base
    out[:, PROGRESS_COLUMN] = np.asarray(progress).reshape(len(rows))
    return out


def from_model_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[T, 30]`` model rows -> float32 ``[T, 20]`` arm rows, ``[T, 9]`` base poses and
    ``[T]`` progress."""
    if rows.ndim != 2 or rows.shape[1] != ACTION_DIM:
        raise ValueError(f"model rows must be [T, {ACTION_DIM}], got {rows.shape}")
    arms = np.array(rows[:, :ARM_ROW_DIM], dtype=np.float32)
    base = np.array(rows[:, BASE_COLUMNS], dtype=np.float32)
    return arms, base, np.array(rows[:, PROGRESS_COLUMN], dtype=np.float32)


class Normalizer:
    """Affine maps between raw and normalized model units.

    Raw to normalized is ``clamp(x * scale + offset, -5, 5)``. Normalized to raw is
    ``(x - offset) / scale``, with no clamp. The action map has one row per chunk position
    (``[32, 30]``) or one row for all positions (``[30]``).
    """

    KEYS = ("action.scale", "action.offset", "proprio.scale", "proprio.offset")

    def __init__(
        self,
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        proprio_scale: torch.Tensor,
        proprio_offset: torch.Tensor,
    ) -> None:
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device="cpu")
        self.action_offset = torch.as_tensor(action_offset, dtype=torch.float32, device="cpu")
        self.proprio_scale = torch.as_tensor(proprio_scale, dtype=torch.float32, device="cpu")
        self.proprio_offset = torch.as_tensor(proprio_offset, dtype=torch.float32, device="cpu")
        if self.action_scale.shape not in ((ACTION_DIM,), (CHUNK_SIZE, ACTION_DIM)):
            raise ValueError(
                f"action scale must be [30] or [32, 30], got {self.action_scale.shape}"
            )
        if self.proprio_scale.shape != (PROPRIO_DIM,):
            raise ValueError(f"proprio scale must be [58], got {self.proprio_scale.shape}")
        for name in ("action", "proprio"):
            scale, offset = getattr(self, f"{name}_scale"), getattr(self, f"{name}_offset")
            if scale.shape != offset.shape:
                raise ValueError(f"{name} scale {scale.shape} and offset {offset.shape} differ")

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, torch.Tensor], prefix: str = "normalizer."
    ) -> "Normalizer":
        """Read ``<prefix>action.scale``, ``action.offset``, ``proprio.scale`` and
        ``proprio.offset``."""
        missing = [prefix + key for key in cls.KEYS if prefix + key not in state_dict]
        if missing:
            raise KeyError(f"normalizer tensors missing from the state dict: {missing}")
        return cls(*(state_dict[prefix + key] for key in cls.KEYS))

    def state_dict(self, prefix: str = "normalizer.") -> dict[str, torch.Tensor]:
        values = (self.action_scale, self.action_offset, self.proprio_scale, self.proprio_offset)
        return {prefix + key: value for key, value in zip(self.KEYS, values, strict=True)}

    def _action_affine(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.action_scale.to(device=actions.device, dtype=actions.dtype)
        offset = self.action_offset.to(device=actions.device, dtype=actions.dtype)
        if scale.ndim == 1:
            return scale, offset
        steps = int(actions.shape[-2])
        if steps > scale.shape[0]:
            raise ValueError(f"{steps} chunk positions, the action map holds {scale.shape[0]}")
        return scale[:steps], offset[:steps]

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        scale = self.proprio_scale.to(device=proprio.device, dtype=proprio.dtype)
        offset = self.proprio_offset.to(device=proprio.device, dtype=proprio.dtype)
        return torch.clamp(proprio * scale + offset, -NORMALIZER_CLAMP, NORMALIZER_CLAMP)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """``[..., T, 30]`` raw rows -> normalized rows. Row ``t`` is chunk position ``t``."""
        scale, offset = self._action_affine(actions)
        return torch.clamp(actions * scale + offset, -NORMALIZER_CLAMP, NORMALIZER_CLAMP)

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scale, offset = self._action_affine(actions)
        return (actions - offset) / scale


def prefix_rows(prefix: ActionChunk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A prefix -> float32 absolute ``[d, 20]`` arm rows, ``[d, 9]`` base poses and ``[d]``
    progress. Without ``prefix.left`` the left arm is the parked placeholder."""
    rows = np.zeros((len(prefix), ARM_ROW_DIM), dtype=np.float32)
    for arm, start, grip in zip(ARMS, POSE_COLUMNS, GRIPPER_COLUMNS, strict=True):
        targets = getattr(prefix, arm)
        if targets is None:
            rows[:, start : start + POSE_DIM] = IDENTITY_POSE
            rows[:, grip] = PARKED_GRIPPER
            continue
        rows[:, start : start + POSE_DIM] = matrix_to_pose(targets.pose)
        rows[:, grip] = targets.gripper
    base = matrix_to_pose(floor_pose(prefix.base_pose)).astype(np.float32)
    return rows, base, prefix.progress


def encode_prefix(
    rows: np.ndarray,
    base: np.ndarray | None,
    progress: np.ndarray,
    reference: np.ndarray,
    base_reference: np.ndarray,
    normalizer: Normalizer,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Absolute ``[d, 20]`` arm rows, ``[d, 9]`` base poses and ``[d]`` progress -> the
    model prefix, float32 ``[1, d, 30]`` normalized in the relative frame, and the relative
    arm rows and base poses. A ``None`` base holds still."""
    relative = to_relative(rows, reference)
    if base is None:
        relative_base = np.tile(IDENTITY_POSE, (len(rows), 1))
    else:
        relative_base = relative_poses(base, base_reference)
    model_rows = torch.from_numpy(to_model_rows(relative, relative_base, progress))[None]
    return normalizer.normalize_actions(model_rows), relative, relative_base


def decode_actions(
    actions: torch.Tensor,
    reference: np.ndarray,
    normalizer: Normalizer,
    timestamp: float,
    base_reference: np.ndarray = IDENTITY_POSE,
    prefix: tuple[np.ndarray, np.ndarray] | None = None,
    single_arm: bool = False,
) -> ActionChunk:
    """Normalized model actions ``[1, 32, 30]`` -> the action chunk for ``timestamp``.

    ``prefix`` is the relative arm rows and base poses from ``encode_prefix``. They replace
    the first rows of the chunk, so the robot gets back exactly the rows it was given.
    """
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != ACTION_DIM:
        raise ValueError(f"actions must be [1, T, {ACTION_DIM}], got {tuple(actions.shape)}")
    raw = normalizer.unnormalize_actions(actions.to(torch.float32))
    rows, base, progress = from_model_rows(raw[0].to(device="cpu", dtype=torch.float32).numpy())
    if prefix is not None:
        rows[: len(prefix[0])] = prefix[0]
        base[: len(prefix[1])] = prefix[1]
    rows = to_absolute(rows, reference)
    rows[:, GRIPPER_COLUMNS] = np.clip(rows[:, GRIPPER_COLUMNS], 0.0, 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    base = relative_poses(base, base_reference, inverse=False)
    arms = {
        arm: ArmTargets(_matrices(rows[:, start : start + POSE_DIM]), rows[:, grip])
        for arm, start, grip in zip(ARMS, POSE_COLUMNS, GRIPPER_COLUMNS, strict=True)
    }
    times = float(timestamp) + (np.arange(len(rows), dtype=np.float64) + 1.0) * DATA_STEP_S
    left = None if single_arm else arms["left"]
    return ActionChunk(times, arms["right"], left, floor_pose(_matrices(base)), progress)


def _matrices(poses: np.ndarray) -> np.ndarray:
    """``[T, 9]`` poses -> float64 ``[T, 4, 4]``."""
    mats = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    mats[:, :3, :3] = rot6d_to_matrix(poses[:, 3:9])
    mats[:, :3, 3] = poses[:, :3]
    return mats
