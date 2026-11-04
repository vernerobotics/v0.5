# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.

import threading

import numpy as np
import pytest
import torch
from torch import nn

from verne import ActionChunk, ArmState, Observation
from verne.model import VideoEncoder, WorldActionModel
from verne.policy import Policy
from verne.transforms import Normalizer

TINY_CONFIG = {
    "num_heads": 2,
    "head_dim": 32,
    "eps": 1e-6,
    "text_dim": 16,
    "text_len": 8,
    "proprio_dim": 58,
    "freq_dim": 16,
    "latent_channels": 48,
    "patch_size": [1, 2, 2],
    "video_dim": 64,
    "video_ffn_dim": 128,
    "paired_video_blocks": [0, 1, 3],
    "action_dim": 30,
    "chunk_size": 32,
    "action_hidden_dim": 48,
    "action_ffn_dim": 96,
    "num_past_frames": 16,
    "past_frame_stride": 8,
    "memory_schedule": [[2, 1, 2], [2, 1, 4], [4, 2, 4], [8, 4, 4]],
    "sigma_shift": 1.5,
    "num_train_timesteps": 1000,
    "max_prefix_len": 12,
    "progress_output": True,
}


class StubTextEncoder(nn.Module):
    """Zero text rows of the tiny width. Counts the prompts it encodes."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, prompts: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        text = torch.zeros(1, TINY_CONFIG["text_len"], TINY_CONFIG["text_dim"])
        return text, torch.ones(1, TINY_CONFIG["text_len"], dtype=torch.bool)


def tiny_policy(**config) -> Policy:
    torch.manual_seed(0)
    model = WorldActionModel.from_config({**TINY_CONFIG, **config}).eval()
    model.vae = VideoEncoder(dim=16).eval()
    normalizer = Normalizer(
        action_scale=torch.ones(32, 30),
        action_offset=torch.zeros(32, 30),
        proprio_scale=torch.ones(58),
        proprio_offset=torch.zeros(58),
    )
    return Policy(
        model,
        StubTextEncoder(),
        normalizer,
        prompt_template="{instruction}",
        num_steps=4,
        text_encoder_on_gpu=True,
    )


def arm(x: float, gripper: float) -> ArmState:
    pose = np.eye(4)
    pose[:3, 3] = [x, 0.1, 0.3]
    return ArmState(pose, gripper)


def observation(t: float, single_arm: bool = False) -> Observation:
    rng = np.random.default_rng(int(t * 1000))
    images = {
        k: rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for k in ("camera_1", "camera_2")
    }
    left = None
    if not single_arm:
        images["camera_0"] = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        left = arm(-0.4, 0.5)
    return Observation(t, "pick up the cup", images, arm(0.4 + 0.01 * t, 0.8), left)


def run(policy: Policy, seed: int, single_arm: bool = False) -> ActionChunk:
    policy.reset(seed=seed)
    for i in range(3):
        policy.observe(observation(i / 30, single_arm))
    return policy.infer(observation(3 / 30, single_arm))


def next_prefix(chunk: ActionChunk, t: float, d: int) -> ActionChunk:
    return chunk.at(t + np.arange(1, d + 1) / 30)


def test_infer_returns_a_chunk_at_30_hz():
    out = run(tiny_policy(), seed=0)
    for targets in (out.left, out.right):
        assert targets.pose.shape == (32, 4, 4)
        assert targets.gripper.shape == (32,)
        assert np.allclose(targets.pose[:, 3], [0, 0, 0, 1])
    assert out.progress.shape == (32,)
    assert np.allclose(out.timestamps, 3 / 30 + (np.arange(32) + 1) / 30)


def test_same_seed_gives_the_same_chunk():
    policy = tiny_policy()
    first, second = run(policy, seed=3), run(policy, seed=3)
    assert np.array_equal(first.right.pose, second.right.pose)
    assert not np.array_equal(first.right.pose, run(policy, seed=4).right.pose)


def test_prefix_rows_come_back_unchanged():
    policy = tiny_policy()
    out = run(policy, seed=0)
    prefix = next_prefix(out, 4 / 30, 5)
    nxt = policy.infer(observation(4 / 30), prefix)
    for got, given in ((nxt.left, prefix.left), (nxt.right, prefix.right)):
        np.testing.assert_allclose(got.pose[:5], given.pose, atol=1e-5)
        np.testing.assert_allclose(got.gripper[:5], given.gripper, atol=1e-6)


def test_each_instruction_is_encoded_once():
    policy = tiny_policy()
    policy.encode_instructions(["pick up the cup", "open the drawer", "pick up the cup"])
    assert policy.text_encoder.calls == 2
    run(policy, seed=0)
    policy.infer(observation(4 / 30))
    assert policy.text_encoder.calls == 2


def test_observe_runs_while_infer_plans():
    policy = tiny_policy()
    for i in range(3):
        policy.observe(observation(i / 30))
    planning, observed = threading.Event(), threading.Event()
    infer_action = policy.model.infer_action

    def slow_infer_action(**kwargs):
        planning.set()
        assert observed.wait(timeout=10)
        return infer_action(**kwargs)

    def camera():
        planning.wait(timeout=10)
        policy.observe(observation(4 / 30))
        observed.set()

    policy.model.infer_action = slow_infer_action
    thread = threading.Thread(target=camera)
    thread.start()
    policy.infer(observation(3 / 30))
    thread.join()
    policy.model.infer_action = infer_action
    policy.infer(observation(5 / 30))


def test_single_arm_reads_and_returns_the_right_arm_only():
    out = run(tiny_policy(), seed=0, single_arm=True)
    assert out.left is None
    assert out.right.pose.shape == (32, 4, 4)


def test_a_config_that_differs_from_the_package_is_rejected():
    with pytest.raises(ValueError, match="past_frame_stride"):
        tiny_policy(past_frame_stride=4)


def test_episode_keeps_its_arms():
    policy = tiny_policy()
    policy.observe(observation(0.0, single_arm=True))
    with pytest.raises(ValueError, match="same arms"):
        policy.observe(observation(1 / 30))


def test_a_rejected_call_leaves_the_policy_unchanged():
    policy = tiny_policy()
    out = run(policy, seed=0)
    late = out.at(4 / 30 + np.arange(2, 8) / 30)
    long = next_prefix(out, 4 / 30, 13)
    for prefix in (late, long):
        with pytest.raises(ValueError):
            policy.infer(observation(4 / 30), prefix)
    policy.infer(observation(4 / 30), next_prefix(out, 4 / 30, 6))


def floor(x: float, y: float, yaw: float) -> np.ndarray:
    out = np.eye(4)
    out[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    out[:2, 3] = [x, y]
    return out


def test_a_base_moving_under_still_hands_reads_as_a_static_base():
    left, right = arm(-0.4, 0.5).pose, arm(0.4, 0.8).pose

    def plan(base_at) -> ActionChunk:
        policy = tiny_policy()
        for i in range(4):
            base = base_at(i / 30)
            to_robot = np.linalg.inv(base)
            obs = Observation(
                i / 30,
                "pick",
                {},
                ArmState(to_robot @ right, 0.8),
                ArmState(to_robot @ left, 0.5),
                base,
            )
            if i < 3:
                policy.observe(obs)
        return policy.infer(obs)

    static = plan(lambda t: np.eye(4))
    moving = plan(lambda t: floor(0.3 * t, -0.2 * t, 0.5 * t))
    np.testing.assert_allclose(moving.left.pose, static.left.pose, atol=1e-5)
    np.testing.assert_allclose(moving.right.pose, static.right.pose, atol=1e-5)


def test_chunk_at_gives_rows_and_interpolates_between_them():
    out = run(tiny_policy(), seed=0)
    rows = out.at(out.timestamps[2:5])
    np.testing.assert_allclose(rows.right.pose, out.right.pose[2:5], atol=1e-6)
    np.testing.assert_allclose(rows.left.gripper, out.left.gripper[2:5], atol=1e-6)
    mid = out.at(out.timestamps[7] + 0.5 / 30)
    expected = 0.5 * (out.right.pose[7, :3, 3] + out.right.pose[8, :3, 3])
    np.testing.assert_allclose(mid.right.pose[0, :3, 3], expected, atol=1e-6)
    rot = mid.right.pose[0, :3, :3]
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)
    late = out.at(out.timestamps[-1] + 1.0)
    np.testing.assert_allclose(late.right.pose[0], out.right.pose[-1], atol=1e-6)
