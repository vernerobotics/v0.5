# Copyright (c) 2026 Verne Labs, Inc.
# Licensed under the Verne V0.5 Model License. See LICENSE.
"""The RoboTwin episode loop: send each observation, run each chunk of joint targets."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for i, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
            if i < len(actions) - 1:
                model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
