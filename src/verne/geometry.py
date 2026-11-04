# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Rotations and poses: 6D rotations, nearest rotations, and 4 x 4 and 9-value poses.

A 9-value pose holds the position, then the first two rows of the rotation matrix.
"""

import numpy as np


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """``[..., 6]`` -> ``[..., 3, 3]``. The 6D vector holds the first two rows of the
    rotation matrix. Gram-Schmidt in the input dtype."""
    row0 = _unit(d6[..., :3])
    row1 = d6[..., 3:]
    row1 = _unit(row1 - np.sum(row0 * row1, axis=-1, keepdims=True) * row0)
    return np.stack((row0, row1, np.cross(row0, row1, axis=-1)), axis=-2)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(v, axis=-1), 1e-12)
    return v / np.expand_dims(norm, -1)


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """``[..., 3, 3]`` -> ``[..., 6]``, the first two rows."""
    return mat[..., :2, :].copy().reshape(mat.shape[:-2] + (6,))


def nearest_rotation(mat: np.ndarray) -> np.ndarray:
    """``[..., 3, 3]`` -> the nearest rotation matrix, float64.

    A matrix that is not orthogonal to 1e-12 is replaced by ``U @ Vt`` of its SVD. The
    result then goes through a unit quaternion (Markley's method) and back.
    """
    m = np.array(mat, dtype=np.float64)
    if np.any(np.linalg.det(m) <= 0):
        raise ValueError("a rotation matrix must have a positive determinant")
    gram = m @ np.swapaxes(m, -1, -2)
    skewed = ~np.all(np.isclose(gram, np.eye(3), rtol=1e-5, atol=1e-12), axis=(-2, -1))
    if skewed.any():
        u, _, vt = np.linalg.svd(m[skewed], full_matrices=False)
        m[skewed] = u @ vt
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    choice = np.argmax(np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], trace], -1), -1)
    cases = (
        (1 - trace + 2 * m[..., 0, 0], m[..., 1, 0] + m[..., 0, 1],
         m[..., 2, 0] + m[..., 0, 2], m[..., 2, 1] - m[..., 1, 2]),
        (m[..., 1, 0] + m[..., 0, 1], 1 - trace + 2 * m[..., 1, 1],
         m[..., 2, 1] + m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0]),
        (m[..., 2, 0] + m[..., 0, 2], m[..., 2, 1] + m[..., 1, 2],
         1 - trace + 2 * m[..., 2, 2], m[..., 1, 0] - m[..., 0, 1]),
        (m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0],
         m[..., 1, 0] - m[..., 0, 1], 1 + trace),
    )  # fmt: skip
    quat = np.empty(m.shape[:-2] + (4,))
    for index, case in enumerate(cases):
        quat = np.where((choice == index)[..., None], np.stack(case, axis=-1), quat)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    elements = (
        x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2,
    )  # fmt: skip
    return np.stack(elements, axis=-1).reshape(m.shape[:-2] + (3, 3))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """``[..., 9]`` pose vectors -> float64 ``[..., 4, 4]``. The rotation is computed in
    the input dtype."""
    shape = pose.shape[:-1]
    rot = rot6d_to_matrix(pose[..., 3:9].reshape(-1, 6)).reshape(*shape, 3, 3)
    mat = np.zeros(shape + (4, 4))
    mat[..., :3, :3] = rot
    mat[..., :3, 3] = pose[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def matrix_to_pose(mat: np.ndarray) -> np.ndarray:
    """``[..., 4, 4]`` -> ``[..., 9]`` pose vectors of the nearest rotation."""
    shape = mat.shape[:-2]
    rot = nearest_rotation(mat[..., :3, :3].reshape(-1, 3, 3))
    return np.concatenate([mat[..., :3, 3], matrix_to_rot6d(rot).reshape(*shape, 6)], axis=-1)


def floor_pose(mat: np.ndarray) -> np.ndarray:
    """``[..., 4, 4]`` base poses -> float64 poses on the floor: the same x, y and yaw,
    z = 0, no roll or pitch."""
    mat = np.asarray(mat, dtype=np.float64)
    yaw = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
    out = np.zeros(mat.shape)
    out[..., 0, 0] = out[..., 1, 1] = np.cos(yaw)
    out[..., 1, 0] = np.sin(yaw)
    out[..., 0, 1] = -out[..., 1, 0]
    out[..., 2, 2] = out[..., 3, 3] = 1.0
    out[..., :2, 3] = mat[..., :2, 3]
    return out
