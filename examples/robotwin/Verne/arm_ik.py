# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Inverse kinematics for the two ARX5 arms of the RoboTwin 2.0 aloha-agilex robot.

Each solve starts at the arm's current joints, so the answer stays on the arm's current
branch. It runs damped least-squares steps on the arm Jacobian. A box keeps every step
inside the joint limits and within `max_step` of the start joints. A small cost pushes a
joint away from a limit. The damping grows near a singularity and after a step that does
not lower the error. A target out of reach gives the nearest pose on the current branch.
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

# Each joint: its origin in the parent link (xyz, rpy) and its axis. From the aloha-agilex
# URDF (arx5_description_isaac.urdf, fl_/fr_joint1..6). The tool frame is link6, the frame
# of RoboTwin's end-effector pose.
CHAIN = (
    ((0.0, 0.0, 0.058), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.025013, 0.00060169, 0.042), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.26396, 0.0044548, 0.0), (-3.1416, 0.0, -0.015928), (0.0, 1.0, 0.0)),
    ((0.246, -0.00025, -0.06), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0015, -0.0855), (0.0, 0.0, -0.015928), (0.0, 0.0, 1.0)),
    ((0.03095, 0.0, 0.0855), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
# Joint limits, deg. Joints 2 to 5 stop where the simulated arm stops. Joints 1 and 6 can
# turn more than one full turn.
LIMITS = np.radians(
    [
        [-425.0, 330.0],
        [-14.0, 200.0],
        [-3.0, 210.0],
        [-105.0, 101.0],
        [-90.0, 90.0],
        [-365.0, 255.0],
    ]
)
# The robot root in the sim world (position, quaternion w first), then the base of each arm
# on the robot (xyz, rpy).
ROOT = ((0.0, -0.65, 0.0), (0.707, 0.0, 0.0, 0.707))
BASES = {
    "left": ((0.2305, 0.297, 0.782), (0.0, 0.0, 0.02)),
    "right": ((0.2315, -0.3063, 0.781), (0.0, 0.0, 0.01)),
}
# Joint speed limit, rad/s.
MAX_SPEED = 5.0

ITERATIONS = 30
CONVERGED_RAD = 1e-6
DAMPING_MIN = 1e-3
DAMPING_MAX = 0.05
SINGULAR = 0.05
LIMIT_BAND = 0.1
LIMIT_WEIGHT = 1e-3


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _transform(rotation: np.ndarray, position) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = position
    return out


def _origin(xyz, rpy) -> np.ndarray:
    return _transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


class ArmIK:
    """One arm: forward kinematics, its Jacobian, and the solve."""

    def __init__(self, base: np.ndarray, max_step: float):
        self._origins = [_origin(xyz, rpy) for xyz, rpy, _ in CHAIN]
        self._axes = [np.array(axis) for _, _, axis in CHAIN]
        self.base = base
        self.max_step = max_step

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The tool pose (4x4, world) and the 6xN world Jacobian [linear; angular]."""
        pose = self.base.copy()
        points, axes = [], []
        for origin, axis, angle in zip(self._origins, self._axes, q, strict=True):
            pose = pose @ origin
            points.append(pose[:3, 3].copy())
            axes.append(pose[:3, :3] @ axis)
            pose[:3, :3] = pose[:3, :3] @ _rotation(axis, angle)
        axes = np.array(axes)
        linear = np.cross(axes, pose[:3, 3] - np.array(points))
        return pose, np.vstack([linear.T, axes.T])

    def _residual(self, q: np.ndarray, target: np.ndarray):
        pose, jac = self.fk(q)
        err = np.concatenate(
            [
                target[:3, 3] - pose[:3, 3],
                Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec(),
            ]
        )
        push = np.clip(q, LIMITS[:, 0] + LIMIT_BAND, LIMITS[:, 1] - LIMIT_BAND) - q
        return err, jac, push, err @ err + LIMIT_WEIGHT * (push @ push)

    def solve(self, q_start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """The joints that reach the world pose `target` (4x4), from `q_start`."""
        q = np.asarray(q_start, dtype=np.float64).copy()
        # A joint past a limit may stay there, but it never moves further out.
        box_lo = np.maximum(np.minimum(LIMITS[:, 0], q), q - self.max_step)
        box_hi = np.minimum(np.maximum(LIMITS[:, 1], q), q + self.max_step)
        eye = np.eye(len(q))
        err, jac, push, cost = self._residual(q, target)
        trust = DAMPING_MIN
        for _ in range(ITERATIONS):
            sigma = np.linalg.svd(jac, compute_uv=False)[-1]
            damping = trust + DAMPING_MAX * max(0.0, 1.0 - (sigma / SINGULAR) ** 2)
            soft = np.sqrt(LIMIT_WEIGHT) * (push != 0.0)
            a = np.vstack([jac, np.sqrt(damping) * eye, np.diag(soft)])
            b = np.concatenate([err, np.zeros(len(q)), soft * push])
            step = lsq_linear(a, b, bounds=(box_lo - q, box_hi - q), method="bvls").x
            trial = np.clip(q + step, box_lo, box_hi)
            t_err, t_jac, t_push, t_cost = self._residual(trial, target)
            # Keep a step only if it lowers the cost.
            if t_cost < cost:
                q, err, jac, push, cost = trial, t_err, t_jac, t_push, t_cost
                trust = max(trust / 10.0, DAMPING_MIN)
                if np.abs(step).max() < CONVERGED_RAD:
                    break
            else:
                trust *= 10.0
        return q


def aloha_arms(step_s: float) -> dict[str, ArmIK]:
    """Both arms, for a control step of `step_s` seconds."""
    root = _transform(Rotation.from_quat(ROOT[1], scalar_first=True).as_matrix(), ROOT[0])
    return {side: ArmIK(root @ _origin(*base), MAX_SPEED * step_s) for side, base in BASES.items()}
