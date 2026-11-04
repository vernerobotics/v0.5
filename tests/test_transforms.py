# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.

import numpy as np
import pytest
import torch

from verne import ActionChunk, ArmState, ArmTargets, Observation
from verne import geometry as G
from verne import transforms as T
from verne.memory_bank import (
    HISTORY_S,
    NUM_SLOTS,
    SLOT_S,
    TICK_S,
    MemoryBank,
    ObservationHistory,
)
from verne.model.text import Tokenizer

RNG = np.random.default_rng(0)


def random_rotations(n: int) -> np.ndarray:
    q, _ = np.linalg.qr(RNG.normal(size=(n, 3, 3)))
    return q * np.sign(np.linalg.det(q))[:, None, None]


def random_poses(n: int) -> np.ndarray:
    mats = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    mats[:, :3, :3] = random_rotations(n)
    mats[:, :3, 3] = RNG.uniform(-0.5, 0.5, size=(n, 3))
    return mats


def random_floor_poses(n: int) -> np.ndarray:
    x, y, yaw = RNG.uniform(-1.0, 1.0, size=(3, n))
    mats = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    mats[:, 0, 0] = mats[:, 1, 1] = np.cos(yaw)
    mats[:, 1, 0], mats[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
    mats[:, 0, 3], mats[:, 1, 3] = x, y
    return mats


def identity_normalizer() -> T.Normalizer:
    return T.Normalizer(
        torch.ones(T.CHUNK_SIZE, T.ACTION_DIM),
        torch.zeros(T.CHUNK_SIZE, T.ACTION_DIM),
        torch.ones(T.PROPRIO_DIM),
        torch.zeros(T.PROPRIO_DIM),
    )


def observation(time: float, left: np.ndarray | None, right: np.ndarray, **kwargs) -> Observation:
    image = np.full((48, 64, 3), int(time * 30) % 256, dtype=np.uint8)
    images = {"camera_1": image, "camera_2": image}
    if left is not None:
        images["camera_0"] = image
        left = ArmState(left, 0.25)
    return Observation(time, "pick up the cup", images, ArmState(right, 0.75), left, **kwargs)


def targets(poses: np.ndarray, gripper: float) -> ArmTargets:
    return ArmTargets(poses, np.full(len(poses), gripper))


def test_rotation_round_trips():
    rots = random_rotations(50)
    np.testing.assert_allclose(G.rot6d_to_matrix(G.matrix_to_rot6d(rots)), rots, atol=1e-12)
    np.testing.assert_allclose(G.nearest_rotation(rots), rots, atol=1e-12)
    skewed = G.nearest_rotation(rots + RNG.normal(scale=1e-3, size=rots.shape))
    np.testing.assert_allclose(
        skewed @ np.swapaxes(skewed, 1, 2), np.broadcast_to(np.eye(3), rots.shape), atol=1e-12
    )
    poses = random_poses(20)
    np.testing.assert_allclose(G.pose_to_matrix(G.matrix_to_pose(poses)), poses, atol=1e-12)
    with pytest.raises(ValueError):
        G.nearest_rotation(np.diag([1.0, 1.0, -1.0]))


def test_resize_image():
    image = RNG.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    np.testing.assert_array_equal(T.resize_image(image, 64, 80), image)
    assert T.resize_image(image, 256, 256).shape == (256, 256, 3)
    flat = np.full((90, 120, 3), 77, dtype=np.uint8)
    np.testing.assert_array_equal(T.resize_image(flat, 256, 256), 77)
    # An exact 2x shrink is the 2 x 2 mean, rounded half up.
    half = T.resize_image(image, 32, 40).astype(int)
    blocks = image.astype(int).reshape(32, 2, 40, 2, 3).sum(axis=(1, 3))
    np.testing.assert_array_equal(half, (blocks + 2) // 4)


def test_canvas():
    images = T.prepare_images({"camera_1": RNG.integers(0, 256, (480, 640, 3), dtype=np.uint8)})
    canvas = T.build_canvas(images)
    assert canvas.shape == (3, 256, 768) and canvas.dtype == torch.float32
    assert torch.all(canvas[:, :, :256] == -1.0) and torch.all(canvas[:, :, 512:] == -1.0)
    expected = torch.from_numpy(images["camera_1"]).permute(2, 0, 1).float() / 255.0
    torch.testing.assert_close(canvas[:, :, 256:512], (expected - 0.5) / 0.5, rtol=0, atol=1e-6)


def test_relative_round_trip():
    reference = np.concatenate([G.matrix_to_pose(random_poses(1))[0], [0.5]] * 2).astype(np.float32)
    rows = np.concatenate([G.matrix_to_pose(random_poses(8)), RNG.uniform(size=(8, 1))] * 2, axis=1)
    rows = rows.astype(np.float32)
    back = T.to_absolute(T.to_relative(rows, reference), reference)
    np.testing.assert_allclose(back, rows, atol=1e-5)


def test_proprio():
    poses = np.stack(
        [[T.orthonormal_pose(T.pose_vector(p)) for p in random_poses(2)] for _ in range(2)]
    )
    grippers = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    proprio = T.build_proprio(poses, grippers, shared_frame=True)
    assert proprio.shape == (T.PROPRIO_DIM,) and proprio.dtype == np.float32
    previous, current = proprio[: T.STATE_ROW_DIM], proprio[T.STATE_ROW_DIM :]
    np.testing.assert_allclose(current[0:9], T.IDENTITY_POSE, atol=1e-6)
    np.testing.assert_allclose(current[10:19], T.IDENTITY_POSE, atol=1e-6)
    np.testing.assert_array_equal(previous[[9, 19]], grippers[0])
    np.testing.assert_array_equal(current[[9, 19]], grippers[1])
    single = T.build_proprio(poses, grippers, shared_frame=False)
    assert not single[20:29].any() and not single[49:58].any()


def test_normalizer():
    torch.manual_seed(0)
    scale = torch.rand(T.CHUNK_SIZE, T.ACTION_DIM) + 0.5
    offset = torch.randn(T.CHUNK_SIZE, T.ACTION_DIM)
    normalizer = T.Normalizer(
        scale, offset, torch.rand(T.PROPRIO_DIM) + 0.5, torch.zeros(T.PROPRIO_DIM)
    )
    loaded = T.Normalizer.from_state_dict(normalizer.state_dict())
    actions = torch.randn(1, T.CHUNK_SIZE, T.ACTION_DIM)
    torch.testing.assert_close(
        loaded.normalize_actions(loaded.unnormalize_actions(actions)), actions
    )
    prefix = torch.randn(1, 5, T.ACTION_DIM)
    expected = (prefix * scale[:5] + offset[:5]).clamp(-T.NORMALIZER_CLAMP, T.NORMALIZER_CLAMP)
    torch.testing.assert_close(normalizer.normalize_actions(prefix), expected)
    assert (
        normalizer.normalize_proprio(torch.full((1, T.PROPRIO_DIM), 1e3)).max()
        == T.NORMALIZER_CLAMP
    )
    with pytest.raises(KeyError):
        T.Normalizer.from_state_dict({"normalizer.action.scale": scale})


def test_prefix_and_decode():
    normalizer = identity_normalizer()
    poses = random_poses(2)
    reference = T.reference_row(
        np.stack([T.orthonormal_pose(T.pose_vector(p)) for p in poses]),
        np.array([0.3, 0.6], np.float32),
    )
    times = 10.0 + np.arange(1, 5) / 30.0
    left = targets(poses[0] @ random_poses(4), 0.3)
    right = targets(np.repeat(poses[1][None], 4, axis=0), 0.6)
    prefix = ActionChunk(times, right, left, np.broadcast_to(np.eye(4), (4, 4, 4)), np.zeros(4))
    rows, _, progress = T.prefix_rows(prefix)
    model_prefix, *relative = T.encode_prefix(
        rows, None, progress, reference, T.IDENTITY_POSE, normalizer
    )
    assert model_prefix.shape == (1, 4, T.ACTION_DIM)
    np.testing.assert_allclose(model_prefix[0, :, 20:29].numpy(), np.tile(T.IDENTITY_POSE, (4, 1)))

    chunk = torch.zeros(1, T.CHUNK_SIZE, T.ACTION_DIM)
    chunk[0, :, 0:9] = chunk[0, :, 10:19] = chunk[0, :, 20:29] = torch.from_numpy(T.IDENTITY_POSE)
    chunk[0, :4] = model_prefix[0]
    out = T.decode_actions(chunk, reference, normalizer, timestamp=10.0, prefix=relative)
    assert out.left.pose.shape == (T.CHUNK_SIZE, 4, 4) and out.right.gripper.shape == (32,)
    np.testing.assert_allclose(out.left.pose[:4], left.pose, atol=1e-5)
    np.testing.assert_allclose(out.right.gripper[:4], 0.6, atol=1e-6)
    np.testing.assert_allclose(out.base_pose, np.broadcast_to(np.eye(4), (32, 4, 4)))
    np.testing.assert_allclose(out.timestamps, 10.0 + np.arange(1, 33) / 30.0)

    single = T.decode_actions(chunk, reference, normalizer, timestamp=0.0, single_arm=True)
    assert single.left is None
    one_arm = ActionChunk(times, right, None, prefix.base_pose, prefix.progress)
    rows, _, _ = T.prefix_rows(one_arm)
    np.testing.assert_array_equal(rows[:, :9], np.tile(T.IDENTITY_POSE, (4, 1)))
    np.testing.assert_array_equal(rows[:, 9], T.PARKED_GRIPPER)


def test_history_resamples_on_the_30hz_grid():
    history = ObservationHistory()
    left, right = random_poses(2)
    times = 5.0 + np.arange(10) / 20.0  # a 20 Hz stream
    for i, t in enumerate(times):
        moved = right.copy()
        moved[0, 3] += 0.01 * i
        history.add(observation(t, left, moved))
    steps = history.resample()
    assert len(steps) == 22 and steps[-1].time == times[-1]
    np.testing.assert_allclose(np.diff([s.time for s in steps]), 1.0 / 30.0, atol=1e-9)
    # The x position moves 0.01 per 1/20 s, so 1/30 s back it is 0.01 * 20 / 30 lower.
    np.testing.assert_allclose(
        steps[-1].poses[1, 0] - steps[-2].poses[1, 0], 0.01 * 20 / 30, atol=1e-6
    )
    np.testing.assert_allclose(steps[-1].grippers, [0.25, 0.75])
    with pytest.raises(ValueError):
        history.add(observation(times[-1], left, right))


def test_history_resizes_only_the_frames_a_canvas_reads(monkeypatch):
    calls = []
    monkeypatch.setattr(T, "resize_image", lambda image, h, w: calls.append(1) or image[:h, :w])
    history = ObservationHistory()
    left, right = random_poses(2)
    for k in range(5):
        history.add(observation(k / 30, left, right))
    assert not calls
    frames = history.resample()[-1].frames
    first = frames.prepared()
    assert len(calls) == 3 and frames.prepared() is first


def test_history_rejects_a_gap_longer_than_it_holds():
    history = ObservationHistory()
    left, right = random_poses(2)
    history.add(observation(1.0, left, right))
    with pytest.raises(ValueError, match="seconds"):
        history.add(observation(1.0 + HISTORY_S + 0.01, left, right))
    history.add(observation(1.0 + HISTORY_S - 0.01, left, right))


def test_history_single_arm():
    history = ObservationHistory()
    history.add(observation(1.0, None, random_poses(1)[0]))
    step = history.resample()[-1]
    np.testing.assert_array_equal(step.poses[0], T.IDENTITY_POSE)
    assert step.grippers[0] == T.PARKED_GRIPPER and not step.frames.prepared()["camera_0"].any()


def latent(step) -> torch.Tensor:
    return torch.full((1, 1, 1, 1, 1), float(step.time), dtype=torch.float64)


def encode(steps) -> torch.Tensor:
    return torch.cat([latent(s) for s in steps])


def test_memory_bank():
    bank = MemoryBank()
    history = ObservationHistory()
    left, right = random_poses(2)
    t0 = 100.0
    history.add(observation(t0, left, right))
    steps = history.resample()
    past, padded = bank.read(steps, latent(steps[-1]), encode)
    assert past.shape == (1, 1, NUM_SLOTS, 1, 1) and padded.all()
    assert torch.all(past == t0)

    # 20 s of 30 Hz frames with a chunk every 0.3 s: every slot is real and on the grid.
    for k in range(1, 601):
        history.add(observation(t0 + k / 30.0, left, right))
        if k % 9 == 0:
            steps = history.resample()
            past, padded = bank.read(steps, latent(steps[-1]), encode)
    assert not padded.any()
    targets = steps[-1].time - SLOT_S * np.arange(NUM_SLOTS, 0, -1)
    assert np.abs(past.flatten().numpy() - targets).max() <= 0.5 * TICK_S + 1e-9

    bank.reset()
    past, padded = bank.read(steps, latent(steps[-1]), encode)
    assert padded.all() and torch.all(past == steps[-1].time)

    # A gap of 1e9 ticks costs no more than a short one.
    late = [T.Step(s.time + 1e9 * TICK_S, s.frames, s.poses, s.grippers, s.base) for s in steps]
    past, padded = bank.read(late, latent(late[-1]), encode)
    assert past.shape == (1, 1, NUM_SLOTS, 1, 1) and padded[0, 0]


def test_decoded_grippers_and_progress_stay_in_0_to_1():
    reference = T.reference_row(np.tile(T.IDENTITY_POSE.astype(np.float64), (2, 1)), np.ones(2))
    chunk = torch.zeros(1, T.CHUNK_SIZE, T.ACTION_DIM)
    chunk[0, :, 0:9] = chunk[0, :, 10:19] = chunk[0, :, 20:29] = torch.from_numpy(T.IDENTITY_POSE)
    chunk[0, :, [9, 19]] = 1.5
    chunk[0, :, T.PROGRESS_COLUMN] = -0.2
    out = T.decode_actions(chunk, reference, identity_normalizer(), 0.0)
    assert np.all(out.left.gripper == 1.0) and np.all(out.right.gripper == 1.0)
    assert np.all(out.progress == 0.0)


def test_a_prompt_that_fills_every_token_is_rejected():
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.max_len = 4

    def tokens(prompts, **kwargs):
        mask = torch.tensor([[1, 1, 0, 0] if len(p) < 8 else [1, 1, 1, 1] for p in prompts])
        return type("Tokens", (), {"input_ids": torch.zeros_like(mask), "attention_mask": mask})

    tokenizer.tokenizer = tokens
    tokenizer(["short"])
    with pytest.raises(ValueError, match="fill all 4"):
        tokenizer(["short", "a long prompt"])


def test_base_targets_are_in_the_frame_of_the_input_base_pose():
    normalizer = identity_normalizer()
    reference = T.reference_row(np.tile(T.IDENTITY_POSE.astype(np.float64), (2, 1)), np.ones(2))
    base_now = random_floor_poses(1)[0]
    prefix_base = base_now @ random_floor_poses(3)
    rows = np.tile(np.concatenate([T.IDENTITY_POSE, [1.0]]), (3, 2)).astype(np.float32)
    model_prefix, *relative = T.encode_prefix(
        rows, G.matrix_to_pose(prefix_base), np.zeros(3), reference,
        T.pose_vector(base_now), normalizer,
    )  # fmt: skip
    chunk = torch.zeros(1, T.CHUNK_SIZE, T.ACTION_DIM)
    chunk[0, :, 0:9] = chunk[0, :, 10:19] = chunk[0, :, 20:29] = torch.from_numpy(T.IDENTITY_POSE)
    chunk[0, :3] = model_prefix[0]
    out = T.decode_actions(
        chunk, reference, normalizer, 0.0, base_reference=T.pose_vector(base_now), prefix=relative
    )
    np.testing.assert_allclose(out.base_pose[:3], prefix_base, atol=1e-5)
    np.testing.assert_allclose(out.base_pose[3:], np.broadcast_to(base_now, (29, 4, 4)), atol=1e-5)
    np.testing.assert_allclose(out.at(out.timestamps[:3]).base_pose, prefix_base, atol=1e-5)


def test_floor_pose_keeps_x_y_and_yaw():
    base = random_floor_poses(1)[0]
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]]
    tilt[2, 3] = 0.3
    np.testing.assert_allclose(G.floor_pose(base @ tilt), base, atol=1e-12)
