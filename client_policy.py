import dataclasses
import enum
import logging
import time

import os
import pickle

import numpy as np
from client.client import WebsocketPolicyClient
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    host: str = "192.168.3.101"
    port: int = 8060

    env: EnvMode = EnvMode.ALOHA_SIM
    num_steps: int = 10

all_time_actions = None
all_actions = None

def get_act_action(obs: dict, policy: WebsocketPolicyClient, stats: dict, state_dim: int, num_queries: int, 
                   temporal_agg: bool, t: int, max_timesteps: int, camera_names: list) -> np.ndarray:
    global all_time_actions, all_actions
    
    pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = num_queries
    if temporal_agg:
        query_frequency = 1
        num_queries = num_queries

    ### evaluation loop
    if temporal_agg and all_time_actions is None:
        all_time_actions = np.zeros([max_timesteps, max_timesteps+num_queries, state_dim])

    ### process previous timestep to get qpos and image_list
    obs = simulate_aloha_obs()
    qpos_numpy = np.array(obs["observation.state"])
    qpos = pre_process(qpos_numpy)
    qpos = np.expand_dims(qpos, axis=0)

    curr_images = []
    for cam_name in camera_names:
        # curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_image = np.transpose(obs[f"observation.images.{cam_name}"], (2, 0, 1))
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0, dtype=np.float32) / 255.0
    curr_image = np.expand_dims(curr_image, axis=0)

    ### query policy
    if t % query_frequency == 0:
        all_actions = policy.infer({
            "qpos": qpos,
            "image": curr_image,
        })
    if temporal_agg:
        all_time_actions[[t], t:t+num_queries] = all_actions
        actions_for_curr_step = all_time_actions[:, t]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = np.expand_dims(exp_weights, axis=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
    else:
        raw_action = all_actions[:, t % query_frequency]

    ### post-process actions
    raw_action = np.squeeze(raw_action, axis=0)
    action = post_process(raw_action)
    target_qpos = action
    
    return target_qpos


def main(args: Args) -> None:
    temporal_agg: bool = True
    max_timesteps: int = 1000

    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    policy = WebsocketPolicyClient(
        host=args.host,
        port=args.port,
    )
    metadata = policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    stats = metadata["stats"]
    state_dim = metadata["state_dim"]
    num_queries = metadata["policy_config"]["num_queries"]
    camera_names = metadata["camera_names"]

    # Send 1 observation to make sure the model is loaded.
    action = policy.infer(obs_fn())
    # print(action)
    print(action.shape)
    print(action.dtype)

    # start = time.time()
    # for _ in range(args.num_steps):
    #     policy.infer(obs_fn())
    # end = time.time()

    start = time.time()
    for i in range(args.num_steps):
        action = get_act_action(obs_fn(), policy, stats, state_dim, num_queries, temporal_agg, 
                                i, max_timesteps, camera_names)
    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")

def simulate_aloha_obs():
    return {
        "observation.state": np.ones((14, ), dtype=np.float32),
        "observation.images.cam1": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam2": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam3": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
    }

def _random_observation_aloha() -> dict:
    return {
        "qpos": np.ones((1, 14), dtype=np.float32),
        "image": np.random.randn(1, 3, 3, 480, 640).astype(np.float32),
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
