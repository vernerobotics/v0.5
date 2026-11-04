# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""Run Verne V0.5 on the LIBERO benchmark.

The episode protocol is the one openpi uses (examples/libero/main.py): seed 7, trial k
starts from LIBERO's init state k, 10 no-op steps while the objects settle, then the policy
until the task is done or the suite's step limit.

    python examples/libero/main.py --suites libero_spatial --trials 10
"""

import argparse
import json
import pathlib

import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

from verne import ActionChunk, ArmState, Observation, Policy

SEED = 7
RESOLUTION = 256
NUM_STEPS_WAIT = 10
NO_OP = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
# The LIBERO controller runs at 20 Hz.
STEP_S = 0.05
MODEL_STEP_S = 1.0 / 30.0
# The model's tool axes in the LIBERO hand frame: R_model = R_hand @ TOOL_AXES.
TOOL_AXES = np.diag([-1.0, -1.0, 1.0])
# robosuite OSC_POSE: an action of 1 moves the goal 0.05 m or 0.5 rad.
OSC_POS_SCALE_M = 0.05
OSC_ROT_SCALE_RAD = 0.5
# The two finger joints of an open Panda gripper sum to this width, m.
JAW_FULL_OPEN_M = 0.08


def init_states(task) -> np.ndarray:
    """LIBERO's fixed start states of a task. LIBERO's own loader fails on torch 2.6 and later,
    which load files with `weights_only=True` by default."""
    path = (
        pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    return torch.load(path, weights_only=False)


def instruction_text(text: str) -> str:
    """One sentence with a capital first letter and one final period, as in training."""
    text = " ".join(text.split()).rstrip(" .,;")
    return text[:1].upper() + text[1:] + "."


class Controller:
    """Turns the policy's absolute targets into LIBERO's OSC_POSE delta actions.

    The model reads the controller goal in force, not the measured hand pose, because the
    LIBERO training data holds the goal.
    """

    def __init__(self, obs: dict) -> None:
        self.goal_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        self.goal_rot = Rotation.from_quat(obs["robot0_eef_quat"])
        fingers = obs["robot0_gripper_qpos"]
        opening = np.clip((fingers[0] - fingers[1]) / JAW_FULL_OPEN_M, 0.0, 1.0)
        self.jaw = 1.0 if opening >= 0.5 else 0.0

    def state(self) -> ArmState:
        pose = np.eye(4)
        pose[:3, :3] = self.goal_rot.as_matrix() @ TOOL_AXES
        pose[:3, 3] = self.goal_pos
        return ArmState(pose, self.jaw)

    def action(self, obs: dict, target_pose: np.ndarray, target_jaw: float) -> np.ndarray:
        """The OSC_POSE action that moves the goal to the target, from the measured pose."""
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        hand = Rotation.from_quat(obs["robot0_eef_quat"])
        target_rot = Rotation.from_matrix(target_pose[:3, :3] @ TOOL_AXES)
        arm = np.concatenate(
            [
                (target_pose[:3, 3] - pos) / OSC_POS_SCALE_M,
                (target_rot * hand.inv()).as_rotvec() / OSC_ROT_SCALE_RAD,
            ]
        )
        arm = np.clip(arm, -1.0, 1.0)
        grip = 1.0 if target_jaw < 0.5 else -1.0
        # The goal the controller sets from this action. A zero rotation keeps the old one.
        self.goal_pos = pos + OSC_POS_SCALE_M * arm[:3]
        if np.any(arm[3:]):
            self.goal_rot = Rotation.from_rotvec(OSC_ROT_SCALE_RAD * arm[3:]) * hand
        self.jaw = 1.0 if grip < 0 else 0.0
        return np.concatenate([arm, [grip]])


def observation(obs: dict, controller: Controller, instruction: str, t: float) -> Observation:
    return Observation(
        timestamp=t,
        instruction=instruction,
        # The renderer gives each image upside down.
        images={
            "camera_1": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
            "camera_2": np.ascontiguousarray(obs["agentview_image"][::-1]),
        },
        right=controller.state(),
    )


def run_episode(env, init_state, policy: Policy, instruction: str, max_steps: int, args, seed):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(NO_OP)
    policy.reset(seed=seed)
    controller = Controller(obs)
    chunk, last_plan, frames = None, 0, []
    for step in range(max_steps):
        t = step * STEP_S
        frames.append(obs["agentview_image"][::-1])
        model_obs = observation(obs, controller, instruction, t)
        if chunk is None or step - last_plan >= args.replan_steps:
            prefix = None
            if chunk is None and args.delay > 0:
                prefix = ActionChunk.hold(model_obs, args.delay)
            elif args.delay > 0:
                prefix = chunk.at(t + MODEL_STEP_S * np.arange(1, args.delay + 1))
            chunk = policy.infer(model_obs, prefix)
            last_plan = step
        else:
            policy.observe(model_obs)
        target = chunk.at(t + STEP_S).right
        action = controller.action(obs, target.pose[0], float(target.gripper[0]))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, frames
    return False, frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="vernerobotics/v0.5-libero")
    parser.add_argument("--suites", nargs="+", default=list(MAX_STEPS), choices=list(MAX_STEPS))
    parser.add_argument("--trials", type=int, default=50, help="trials per task, up to 50")
    parser.add_argument("--replan-steps", type=int, default=5, help="env steps between chunks")
    parser.add_argument(
        "--delay",
        type=int,
        default=6,
        help="rows of the running chunk each new chunk keeps (real-time chunking); 0 = off",
    )
    parser.add_argument("--compile", action="store_true", help="compile the model")
    parser.add_argument("--out", default="results/libero")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out = pathlib.Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    policy = Policy.from_pretrained(args.checkpoint, compile=args.compile)
    np.random.seed(SEED)
    rates = {}
    with open(out / "episodes.jsonl", "w") as log:
        for suite_name in args.suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            successes = 0
            for task_id in range(suite.n_tasks):
                task = suite.get_task(task_id)
                states = init_states(task)
                bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder
                env = OffScreenRenderEnv(
                    bddl_file_name=str(bddl / task.bddl_file),
                    camera_heights=RESOLUTION,
                    camera_widths=RESOLUTION,
                )
                env.seed(SEED)
                for trial in range(args.trials):
                    instruction = instruction_text(task.language)
                    success, frames = run_episode(
                        env,
                        states[trial],
                        policy,
                        instruction,
                        MAX_STEPS[suite_name],
                        args,
                        trial,
                    )
                    successes += success
                    name = f"{suite_name}_{task_id}_{trial}_{'ok' if success else 'fail'}.mp4"
                    imageio.mimwrite(out / "videos" / name, frames, fps=20)
                    row = {"suite": suite_name, "task_id": task_id, "trial": trial}
                    row.update(task=task.language, success=success)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                env.close()
            rates[suite_name] = successes / (suite.n_tasks * args.trials)
            print(f"{suite_name}: {100 * rates[suite_name]:.1f}% success", flush=True)
    print(f"mean: {100 * np.mean(list(rates.values())):.1f}% over {len(rates)} suites")


if __name__ == "__main__":
    main()
