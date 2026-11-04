# LIBERO

This example runs the LIBERO checkpoint on the four LIBERO suites with the episode
protocol of [openpi](https://github.com/Physical-Intelligence/openpi): seed 7, 50 trials
per task, 10 no-op steps at the start, and each suite's step limit.

## Set up

Use the same Python environment as Verne V0.5 (Python 3.10 or newer).

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
pip install -r examples/libero/requirements.txt
export PYTHONPATH=$PWD/LIBERO:$PYTHONPATH
```

The first `import libero` asks where to keep its datasets. Answer `N` to keep the default.

## Run

```bash
export MUJOCO_GL=egl
python examples/libero/main.py --suites libero_spatial --trials 10
```

Leave out `--suites` and `--trials` to run the full benchmark: 4 suites x 10 tasks x 50
trials. The script writes `results/libero/episodes.jsonl` and one video per episode, and
prints the success rate of each suite.

## How it works

- The model sees the wrist camera as `camera_1` and the third-person camera as `camera_2`.
  The left arm is not there, so each observation has `left=None`.
- The model plans 32 poses at 30 Hz. The env steps at 20 Hz. Each step sends the planned
  pose one step ahead as a LIBERO delta action.
- A new chunk starts every 5 env steps (`--replan-steps`). It keeps the next 6 targets of
  the running chunk (`--delay`). The first chunk keeps 6 copies of the current pose: the
  robot holds still while it plans. Use `--delay 0` to plan each chunk from scratch.
