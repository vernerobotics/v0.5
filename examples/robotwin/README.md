# RoboTwin 2.0

This example runs the RoboTwin checkpoint on the
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) benchmark. The `Verne` folder
is a policy for RoboTwin's XPolicyLab harness. RoboTwin runs the sim in its own conda
environment, and the policy runs as a server in the Verne V0.5 environment.

## Set up

1. Install RoboTwin 2.0 by its own instructions, at commit `ea8b2112`, with the XPolicyLab
   submodule. This gives the `RoboTwin` conda environment.
2. Install Verne V0.5 in its own environment (see the main README). Then add the packages
   the policy server needs:

   ```bash
   pip install -r examples/robotwin/requirements.txt
   ```

3. Copy the policy folder into RoboTwin:

   ```bash
   cp -r examples/robotwin/Verne <RoboTwin>/XPolicyLab/policy/Verne
   ```

## Run

From the RoboTwin folder:

```bash
bash XPolicyLab/policy/Verne/eval.sh RoboTwin beat_block_hammer vernerobotics/v0.5-robotwin \
    aloha_agilex joint 0 0 0 <path to the Verne V0.5 environment> RoboTwin
```

The arguments are: the bench, the task, the checkpoint (a Hugging Face repo or a local
folder), the robot, the action type, the eval seed, the GPU of the policy, the GPU of the
sim, the Verne V0.5 environment folder, and the RoboTwin conda environment.

The default runs 100 episodes in the `demo_clean` setting. For the `demo_randomized`
setting, put the sim options in a file, one per line, and point RoboTwin to it:

```bash
printf -- '--task_config\ndemo_randomized\n' > randomized.txt
export ROBOTWIN_EVAL_ARGS_FILE=$PWD/randomized.txt
```

RoboTwin writes the results under `eval_result/`.

## How it works

- The model sees the left wrist camera as `camera_0`, the right wrist camera as
  `camera_1`, and the head camera as `camera_2`. The arm poses are the world poses of each
  arm's last link.
- One env step is 0.06 s. A new chunk starts every 10 steps (`replan_steps` in
  `deploy.yml`). It keeps the next 6 targets of the running chunk (`delay`). The first
  chunk keeps 6 copies of the current pose: the robot holds still while it plans. Set
  `delay: 0` to plan each chunk from scratch.
- The planned poses go to joint targets through the inverse kinematics in `arm_ik.py`. Each
  solve starts at the arm's joint targets of the last step.
