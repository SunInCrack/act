import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, camera_names, norm_stats, is_sim):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = is_sim
        # self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = self.episode_ids[index]
        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']
            original_action_shape = root['/frames/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len - 100)  # 保证其动作序列长度不会比detr的num_queries短，导致维数错误（有点简陋）
            # get observation at start_ts only
            state = root['/frames/state'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/frames/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if self.is_sim:
                action = root['/frames/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/frames/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        state_data = torch.from_numpy(state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        state_data = (state_data - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]

        return image_data, state_data, action_data, is_pad


def get_norm_stats(dataset_dir, episodes):
    all_state_data = []
    all_action_data = []
    for episode_idx in episodes:
        dataset_path = episode_idx
        with h5py.File(dataset_path, 'r') as root:
            state = root['/frames/state'][()]
            action = root['/frames/action'][()]
        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))
    all_state_data = torch.cat(all_state_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize state data
    state_mean = all_state_data.mean(dim=0, keepdim=True)
    state_std = all_state_data.std(dim=0, keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "state_mean": state_mean.numpy().squeeze(), "state_std": state_std.numpy().squeeze(),
             "example_state": state}

    return stats

def collate_fn(batch):
    min_len = min([b[3].shape[0] for b in batch])
    batch = list(zip(*batch))
    image_data = torch.stack(batch[0])
    state_data = torch.stack(batch[1])
    action_data, is_pad = [torch.stack([item[: min_len] for item in items]) for items in batch[2: ]]
    return image_data, state_data, action_data, is_pad

def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val, is_sim):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    long_episodes = []
    for file in os.listdir(dataset_dir):
        with h5py.File(os.path.join(dataset_dir, file), 'r') as f:
            if f["/frames/action"].shape[0] >= 200:
                long_episodes.append(os.path.join(dataset_dir, file))

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(long_episodes)
    train_indices = shuffled_indices[:int(train_ratio * len(long_episodes))]
    val_indices = shuffled_indices[int(train_ratio * len(long_episodes)):]

    # obtain normalization stats for state and action
    norm_stats = get_norm_stats(dataset_dir, shuffled_indices)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, camera_names, norm_stats, is_sim)
    val_dataset = EpisodicDataset(val_indices, camera_names, norm_stats, is_sim)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
