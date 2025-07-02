import os
import dataclasses
import enum
import logging
import socket
import pickle
import json
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import tyro
import argparse
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

from policy import ACTPolicy, CNNMLPPolicy
from server.policy_server import WebsocketPolicyServer

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def main(args) -> None:
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
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
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
            'hdf5_keys': {
                'state': state,
                'action': action,
                'images': images,
            },
            'real_robot': not is_sim,
            'stats': stats
        }
    
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_20000_seed_0.ckpt')
    policy = make_policy(policy_class, config['policy_config'])
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(loading_status)
    # print(next(policy.parameters()).device)
    policy.cuda()
    policy.eval()


    policy_metadata = config

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args["port"],
        metadata=policy_metadata,
    )
    server.serve_forever()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, force=True)
#     main(tyro.cli(Args))

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

    # for server
    parser.add_argument('--port', action='store', type=int, default=8060, help='the port the server is listening to', required=False)
    parser.add_argument('--arm', choices=["left", "right", "both"], default="both", type=str, help="the arms to be used")
    
    main(vars(parser.parse_args()))
