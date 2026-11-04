# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.

import numpy as np
import pytest

from verne import ActionChunk, ArmState, ArmTargets, Observation


def pose(x: float = 0.3) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    out[:3, 3] = [x, 0.1, 0.2]
    return out


def test_a_pose_must_be_a_rigid_transform():
    ArmState(pose().astype(np.float32), 0.5)
    with pytest.raises(ValueError, match="transposed"):
        ArmState(pose().T, 0.5)
    flipped = pose()
    flipped[:3, 1] *= -1
    with pytest.raises(ValueError, match="flipped"):
        ArmState(flipped, 0.5)
    with pytest.raises(ValueError, match="orthonormal"):
        ArmState(pose() @ np.diag([1.1, 1.0, 1.0, 1.0]), 0.5)
    with pytest.raises(ValueError):
        ArmState(np.full((4, 4), np.nan), 0.5)


def test_observation_checks_its_images():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    obs = Observation(0.0, "pick", {"camera_1": image}, ArmState(pose(), 1.0))
    assert obs.images["camera_0"] is None and not obs.images["camera_1"].flags.writeable
    with pytest.raises(KeyError):
        Observation(0.0, "pick", {"camera_3": image}, ArmState(pose(), 1.0))
    with pytest.raises(ValueError, match="uint8"):
        Observation(0.0, "pick", {"camera_1": image.astype(np.float32)}, ArmState(pose(), 1.0))
    with pytest.raises(ValueError, match="no left arm"):
        Observation(0.0, "pick", {"camera_0": image}, ArmState(pose(), 1.0))


def test_hold_repeats_the_current_state():
    obs = Observation(2.0, "pick", {}, ArmState(pose(0.4), 0.7), ArmState(pose(-0.4), 0.2))
    held = ActionChunk.hold(obs, 6)
    np.testing.assert_allclose(held.timestamps, 2.0 + np.arange(1, 7) / 30)
    np.testing.assert_array_equal(held.left.pose, np.broadcast_to(pose(-0.4), (6, 4, 4)))
    np.testing.assert_allclose(held.right.gripper, 0.7)
    np.testing.assert_array_equal(held.base_pose, np.broadcast_to(np.eye(4), (6, 4, 4)))
    base = np.eye(4)
    base[:3, 3] = [1.0, 2.0, 0.0]
    moved = ActionChunk.hold(Observation(2.0, "pick", {}, obs.right, base_pose=base), 1)
    np.testing.assert_allclose(moved.right.pose[0], base @ pose(0.4))


def test_chunk_rows_must_agree():
    poses = np.broadcast_to(pose(), (3, 4, 4))
    with pytest.raises(ValueError, match="rows"):
        ActionChunk(np.arange(3.0), ArmTargets(poses, np.ones(3)), None, poses[:2], np.zeros(3))
    with pytest.raises(ValueError, match="rise"):
        ActionChunk(np.zeros(3), ArmTargets(poses, np.ones(3)), None, poses, np.zeros(3))
