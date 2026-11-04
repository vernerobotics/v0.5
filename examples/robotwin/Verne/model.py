# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Verne V0.5 as an XPolicyLab policy for RoboTwin 2.0.

XPolicyLab calls `update_obs` after every env step and `get_action` when the last chunk of
joint targets has run. Each `get_action` plans a new chunk and turns the next
`replan_steps` poses into joint targets with the arm IK.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from XPolicyLab.model_template import ModelTemplate

from verne import ActionChunk, ArmState, Observation, Policy

from .arm_ik import aloha_arms

# One env step is 15 steps of the 250 Hz sim, the frame period of the RoboTwin data.
STEP_S = 15 / 250
MODEL_STEP_S = 1 / 30
SIDES = ("left", "right")
CAMERAS = {"cam_left_wrist": "camera_0", "cam_right_wrist": "camera_1", "cam_head": "camera_2"}
# The model's tool axes in the RoboTwin link6 frame: R_model = R_link6 @ TOOL_AXES.
TOOL_AXES = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def model_pose(pose7) -> np.ndarray:
    """RoboTwin `[x, y, z, qw, qx, qy, qz]` (link6, world) -> the model's 4x4 pose."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(pose7[3:], scalar_first=True).as_matrix() @ TOOL_AXES
    out[:3, 3] = pose7[:3]
    return out


def link6_pose(pose: np.ndarray) -> np.ndarray:
    """The model's 4x4 pose -> the 4x4 world pose of link6."""
    out = pose.copy()
    out[:3, :3] = pose[:3, :3] @ TOOL_AXES.T
    return out


class Model(ModelTemplate):
    def __init__(self, deploy_cfg):
        super().__init__()
        self.model = Policy.from_pretrained(
            deploy_cfg["checkpoint"], compile=bool(deploy_cfg["compile"])
        )
        self.replan_steps = int(deploy_cfg["replan_steps"])
        self.delay = int(deploy_cfg["delay"])
        self.arms = aloha_arms(STEP_S)
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.model.reset(seed=self.episode)
        self.steps = 0
        self.pending = None
        self.joints = None
        self.chunk = None

    def update_obs(self, obs):
        # Each observation goes to the policy once: by `observe`, or by `infer` when a
        # chunk is planned from it.
        if self.pending is not None:
            self.model.observe(self.pending)
        state = obs["state"]
        arms = {
            side: ArmState(
                model_pose(state[f"{side}_ee_pose"]),
                float(np.ravel(state[f"{side}_ee_joint_state"])[0]),
            )
            for side in SIDES
        }
        self.pending = Observation(
            timestamp=self.steps * STEP_S,
            instruction=obs["instruction"],
            images={slot: obs["vision"][cam]["color"] for cam, slot in CAMERAS.items()},
            right=arms["right"],
            left=arms["left"],
        )
        self.joints = {side: np.asarray(state[f"{side}_arm_joint_state"]) for side in SIDES}
        self.steps += 1

    def get_action(self):
        obs, self.pending = self.pending, None
        t = obs.timestamp
        prefix = None
        if self.chunk is None and self.delay > 0:
            prefix = ActionChunk.hold(obs, self.delay)
        elif self.delay > 0:
            prefix = self.chunk.at(t + MODEL_STEP_S * np.arange(1, self.delay + 1))
        self.chunk = self.model.infer(obs, prefix)

        # Step k ends at t + (k + 1) * STEP_S, so it goes to the planned pose of that time.
        targets = self.chunk.at(t + STEP_S * np.arange(1, self.replan_steps + 1))
        q = dict(self.joints)
        actions = []
        for k in range(self.replan_steps):
            action = {}
            for side in SIDES:
                arm = getattr(targets, side)
                q[side] = self.arms[side].solve(q[side], link6_pose(arm.pose[k]))
                action[f"{side}_arm_joint_state"] = q[side]
                action[f"{side}_ee_joint_state"] = np.clip(arm.gripper[k : k + 1], 0, 1)
            actions.append(action)
        return actions
