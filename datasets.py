# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                #原逻辑只有下面一行：
                #self.index_to_data = pickle.load(f)
                data = pickle.load(f)
                # 兼容两种格式：tuple (index_to_data, goals_index) 或直接的 list
                if isinstance(data, tuple):
                    self.index_to_data, self.goals_index = data
                else:
                    self.index_to_data = data
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )
            
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        遍历所有轨迹，把每个可用的观测点（curr_time）都加入索引。对每个观测点，记录可采样目标的最小/最大距离。
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            
            # 0529 Skip trajectories that are too short to form a valid sample  仅在需要infer scand的时候使用
            if traj_len < self.context_size + self.len_traj_pred:
                print(f"[WARNING] [{self.dataset_name}] Skipping trajectory '{traj_name}': "
                      f"length {traj_len} < context_size({self.context_size}) + len_traj_pred({self.len_traj_pred}) = "
                      f"{self.context_size + self.len_traj_pred}")
                continue
            
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))
            #遍历轨迹，从第 context_size-1 到 traj_len-len_traj_pred 的每个时间点作为观测点（curr_time）
            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            #对每个观测点，随机采样 goals_per_obs 个目标帧，目标帧距离观测点在 [min_dist_cat, max_goal_distance] 范围内。
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index
  
    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            #traj_data[k] = v.astype('float')
            v_arr = np.asarray(v)
            if v_arr.dtype.kind in ('f', 'i', 'u'):
                traj_data[k] = v_arr.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        # Handle datasets where yaw/position elements are arrays (e.g., go_stanford has dtype=object)
        yaw_arr = np.asarray(yaw)
        positions_arr = np.asarray(positions)
        if yaw_arr.dtype == object or positions_arr.dtype == object:
            # Convert to proper float arrays
            yaw_fixed = np.array([float(y[0]) if isinstance(y, np.ndarray) and y.shape == (1,) else float(y) for y in yaw_arr], dtype=np.float64)
            yaw = yaw_fixed

            positions_fixed = np.array([[float(x) for x in p] for p in positions_arr], dtype=np.float64)
            positions = positions_fixed

            # Also fix goal_yaw and goal_pos
            goal_yaw_arr = np.asarray(goal_yaw)
            if isinstance(goal_yaw_arr, np.ndarray) and goal_yaw_arr.shape == (1,):
                goal_yaw = float(goal_yaw_arr[0])
            elif goal_yaw_arr.dtype == object:
                goal_yaw = float(goal_yaw_arr.item())
            else:
                goal_yaw = float(goal_yaw_arr)

            goal_pos_arr = np.asarray(goal_pos)
            if goal_pos_arr.dtype == object:
                # goal_pos_arr is like [[x, y]] or [x, y]
                goal_pos = np.array([[float(v) for v in row] for row in goal_pos_arr], dtype=np.float64)
            else:
                goal_pos = np.asarray(goal_pos, dtype=np.float64)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw_diff = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw_diff.reshape(-1, 1)], axis=-1)
        return actions, goal_pos    

class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

    #用于为模型提供单个训练样本。它通过时间索引加载历史图像帧（Context）和未来目标帧（Goal），并读取相应的位姿（Pose）数据作为标签
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            #f_curr: 当前轨迹的文件夹 ID 或名称。curr_time: 当前观测时刻的时间步索引。min/max_goal_dist: 允许预测的未来时间窗口范围。
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            #在允许的范围内随机采样未来目标偏移量，有goals_per_obs个
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            #计算绝对目标时间
            goal_time = (curr_time + goal_offset).astype('int')
            #将时间偏移量归一化。128 是一个超参数（可能是最大可能的序列长度或经验值），用于将时间输入缩放到模型易于处理的数值范围
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const
            #构建历史上下文时间列表，从curr time往前数context_size帧
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            #将“历史帧”和“未来目标帧”的索引合并成一个列表。元组格式为 (轨迹ID, 时间索引)
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]
            #Image.open: 从磁盘读取图片，self.transform: 进行预处理（Resize, ToTensor, Normalize）。单张图变 $(3, H, W)$，torch.stack: 将列表堆叠为一个 Tensor
            #输入：文件路径列表，输出obs_image (Tensor)，维度为 $(C + K, 3, H, W)$，前C张是历史帧，后K张是未来帧
            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data根据轨迹 ID 读取该段视频对应的所有数值数据
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions从整条轨迹数据中提取出 goal_time 对应的位姿信息
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            #对位姿的前两维（通常是 X, Y 坐标）进行标准化处理（减均值除方差），以便模型训练收敛
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),#x 包含历史帧和目标帧的图像（通常 shape 为 [context_size + goals_per_obs, C, H, W]）。content_size即观测点之前的连续帧
                torch.as_tensor(goal_pos, dtype=torch.float32),#y 未来K个目标点的真实物理位置/位姿，目标帧的位姿（位置和朝向），用于监督模型输出。
                torch.as_tensor(rel_time, dtype=torch.float32),#rel_t 每个目标帧距离观测点的时间间隔（归一化），shape 为 [goals_per_obs]
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)

class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)
  
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1])) # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
        
class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

   
    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))           
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
