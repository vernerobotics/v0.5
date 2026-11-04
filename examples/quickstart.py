# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Load Verne V0.5 and plan a few action chunks from made-up observations.

python examples/quickstart.py --checkpoint vernerobotics/v0.5-base
"""

import argparse
import time

import numpy as np

from verne import ActionChunk, ArmState, Observation, Policy

STEP_S = 1.0 / 30.0
REPLAN_STEPS = 10
DELAY_ROWS = 6


def arm(y: float) -> ArmState:
    pose = np.eye(4)
    # The gripper points forward and level: the tool +z along the robot +x, +x down.
    pose[:3, :3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    pose[:3, 3] = [0.4, y, 0.3]
    return ArmState(pose, gripper=1.0)


def observation(t: float, rng: np.random.Generator) -> Observation:
    return Observation(
        timestamp=t,
        instruction="Pick up the red cup and put it in the sink.",
        images={
            name: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            for name in ("camera_0", "camera_1", "camera_2")
        },
        right=arm(-0.25),
        left=arm(0.25),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="vernerobotics/v0.5-base")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    policy = Policy.from_pretrained(args.checkpoint, compile=args.compile)
    policy.reset(seed=0)
    rng = np.random.default_rng(0)
    chunk = None
    for step in range(5 * REPLAN_STEPS):
        t = step * STEP_S
        obs = observation(t, rng)
        if step % REPLAN_STEPS:
            policy.observe(obs)
            continue
        # Keep the rows the robot runs while the new chunk is planned (real-time chunking).
        # The robot holds still while the first chunk is planned.
        if chunk is None:
            prefix = ActionChunk.hold(obs, DELAY_ROWS)
        else:
            prefix = chunk.at(t + STEP_S * np.arange(1, DELAY_ROWS + 1))
        start = time.perf_counter()
        chunk = policy.infer(obs, prefix)
        took = time.perf_counter() - start
        print(
            f"t={t:.2f}s planned {len(chunk)} rows in {took:.2f}s, "
            f"right hand goal {chunk.right.pose[-1, :3, 3].round(3)}, "
            f"progress {chunk.progress[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
