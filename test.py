import h5py
import os
from utils import load_data
import socket

def print_h5(items):
    for key, item in items:
        if isinstance(item, h5py.Dataset):  # 检查是否为数据集
            print(f"数据集名: {key}, 形状: {item.shape}, 类型: {item.dtype}")
        elif isinstance(item, h5py.Group):  # 如果需要遍历组中的内容
            print(f"组名{key}: ")
            print_h5(item.items())
            print()
 
# path = "datasets/sim_transfer_cube_scripted/episode_0.hdf5"
# path = "datasets/sim_object_placement/episode_134.h5"

dir = '/data/dataset/real/aloha/data/clean_data_without_split/hdf5/pick_put_cube_0624'
episode_len = []

# 打开HDF5文件
for file in os.listdir(dir):
    with h5py.File(os.path.join(dir, file), 'r') as file:
        # 打印所有顶级项（即组和数据集）的名称
        # print("顶级项:")
        # for key in file.keys():
        #     print(key)

        print_h5(file.items())

        # print(file["/frames/action"][0])
        # print(file["/frames/language_instruction"][0])
        # print(file["/frames/observation_images_cam_exterior"][0])
        # print(file["/frames/observation_images_cam_wrist"][0])
        # print(file["/frames/state"][0])
        # print(list(file.attrs))

        # episode_len.append(file["/frames/action"].shape[0])

        episode_len.append(file["/action"].shape[0])
print(sorted(episode_len))

# load_data("datasets/sim_object_placement", ["a"], 1, 1, True)

