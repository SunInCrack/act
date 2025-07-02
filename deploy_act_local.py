import os
import dataclasses
import enum
import logging
import pickle
import time
import json
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import cv2
from einops import rearrange

import tyro
import argparse
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

from policy import ACTPolicy, CNNMLPPolicy


def simulate_aloha_obs():
    return {
        "observation.state": np.ones((14, ), dtype=np.float32),
        "observation.images.cam1": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam2": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam3": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
    }


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def main(args) -> None:
    max_timesteps: int = 500
    temporal_agg: bool = False

    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    state = task_config['state']
    action = task_config['action']
    images = task_config['images']

    # fixed parameters
    with h5py.File(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]), 'r') as f:
        state_dim = f[state].shape[-1]
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        if args['arm'] != 'both':
            assert state_dim % 2 == 0, "It's a sinle arm robot"
            state_dim //= 2
        policy_config = {'lr': args['lr'],
                         'weight_decay': args['weight_decay'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'state_dim': state_dim,
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': args['lr_backbone'],
                         'backbone': args['backbone'],
                         'dropout': args['dropout'],
                         'pre_norm': args['pre_norm'],
                         'position_embedding': args['position_embedding'],
                         'masks': args['masks'],
                         'dilation': args['dilation'],
                         'enc_layers': args['enc_layers'],
                         'dec_layers': args['dec_layers'],
                         'nheads': args['nheads'],
                         'camera_names': camera_names,
                         'arm': args['arm'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': args['lr_backbone'], 'backbone' : args['backbone'], 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if os.path.exists(os.path.join(ckpt_dir, "config.json")):
        with open(os.path.join(ckpt_dir, "config.json"), "r") as json_file:
            config = json.load(json_file)
            config['stats'] = stats
    else:
        config = {
            'num_epochs': num_epochs,
            'ckpt_dir': ckpt_dir,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'lr': args['lr'],
            'policy_class': policy_class,
            'onscreen_render': onscreen_render,
            'policy_config': policy_config,
            'task_name': task_name,
            'seed': args['seed'],
            'temporal_agg': args['temporal_agg'],
            'camera_names': camera_names,
            'real_robot': not is_sim,
            'hdf5_keys': {
                'state': state,
                'action': action,
                'images': images,
            },
            'stats': stats
        }
    
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_20000_seed_0.ckpt')
    policy = make_policy(policy_class, config['policy_config'])
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(loading_status)
    # print(next(policy.parameters()).device)
    policy.cuda()
    policy.eval()

    pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # Start the robot
    if not args['test']:
        from robot.airbot import AIRBOTPlay, _ROBOT_CONFIG
        robot = AIRBOTPlay()

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    with torch.inference_mode():
        start = time.perf_counter()
        for t in range(max_timesteps):
            if args['test']:
                obs = simulate_aloha_obs()
            else:
                obs = robot.capture_observation()

            qpos_numpy = np.array(obs['observation.state'], dtype=np.float32)
            # select the arm
            if args['arm'] == 'left':
                qpos_numpy, _ = np.split(qpos_numpy, 2, axis=-1)
            elif args['arm'] == 'right':
                _, qpos_numpy = np.split(qpos_numpy, 2, axis=-1)

            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).cuda().unsqueeze(0)   # .float()效率极其低下，最好在创建时就转换类型

            curr_images = []
            for cam_name in camera_names:
                curr_image = obs[f"observation.images.{cam_name}"].transpose((2, 0, 1))
                curr_images.append(curr_image)

                if not args['test']:
                    # display the frames
                    view = cv2.putText(obs[f"observation.images.{cam_name}"], f"frame {t}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(cam_name, view)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            curr_image = np.stack(curr_images, axis=0, dtype=np.float32)
            curr_image = torch.from_numpy(curr_image / 255.0).cuda().unsqueeze(0)   # .float()效率极其低下，最好在创建时就转换类型

            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01    # temporal aggregation coefficient
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image)
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)

            # control the robot
            if not args['test']:
                if config['policy_config']['arm'] == 'left':
                    action = np.concate([
                        action,
                        np.array(_ROBOT_CONFIG['start_arm_joint_position'][1])
                    ], 
                    axis=0)
                elif config['policy_config']['arm'] == 'right':
                    action = np.concate([
                        np.array(_ROBOT_CONFIG['start_arm_joint_position'][0]),
                        action
                    ], 
                    axis=0)
                robot.send_action(action)

            # print(action)
            print(f"Average FPS: {(t + 1) / (time.perf_counter() - start)} Hz")
            
        end = time.perf_counter()

        print(f"Total time taken: {end - start:.2f} s")
        print(f"Average inference time: {1000 * (end - start) / max_timesteps:.2f} ms")
    
    if not args['test']:
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=7, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # for deploy
    parser.add_argument('--test', action='store_true', help='test mode or real mode')
    parser.add_argument('--arm', choices=["left", "right", "both"], default="both", type=str, help="the arms to be used")
    
    main(vars(parser.parse_args()))
