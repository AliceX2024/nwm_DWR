#!/usr/bin/env python3
"""Debug script to identify the inhomogeneous shape error"""
import pickle
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from PIL import Image
import yaml
from misc import transform, get_data_path, normalize_data, get_delta_np

# Load config
with open("config/data_config.yaml", "r") as f:
    all_data_config = yaml.safe_load(f)

# Load go_stanford time.pkl
pkl_path = 'data_splits/go_stanford/test/time.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

data_folder = 'data/go_stanford'
dataset_name = 'go_stanford'
context_size = 4
len_traj_pred = 64
metric_waypoint_spacing = all_data_config[dataset_name]['metric_waypoint_spacing']

print(f"Dataset: {dataset_name}")
print(f"Total samples: {len(data)}")
print(f"context_size: {context_size}, len_traj_pred: {len_traj_pred}")
print(f"metric_waypoint_spacing: {metric_waypoint_spacing}")

# ACTION_STATS
ACTION_STATS = {}
for key in all_data_config['action_stats']:
    ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)
print(f"ACTION_STATS: {ACTION_STATS}")

def _get_trajectory(traj_name):
    traj_path = os.path.join(data_folder, traj_name, "traj_data.pkl")
    with open(traj_path, "rb") as f:
        traj_data = pickle.load(f)
    for k, v in traj_data.items():
        v_arr = np.asarray(v)
        if v_arr.dtype == object:
            # Check if each element is a 0-dim array (need to extract scalar)
            sample_elem = np.asarray(v_arr.flat[0])
            if v_arr.ndim == 1:
                v_arr = np.array([float(np.asarray(elem)) for elem in v_arr.flat])
            elif v_arr.ndim == 2 and v_arr.shape[1] == 1:
                v_arr = np.array([float(np.asarray(elem)) for elem in v_arr.flat])
            else:
                v_arr = np.array([[float(np.asarray(elem2)) for elem2 in elem1] for elem1 in v_arr])
        elif v_arr.dtype.kind in ('f', 'i', 'u'):
            v_arr = v_arr.astype('float')
        traj_data[k] = v_arr
    return traj_data

def to_local_coords(positions, curr_pos, curr_yaw):
    def yaw_rotmat(yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError
    return (positions - curr_pos) @ rotmat.T

def angle_difference(yaw1, yaw2):
    diff = np.asarray(yaw2) - np.asarray(yaw1)
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return diff

def _compute_actions(traj_data, curr_time, goal_time):
    start_index = curr_time
    end_index = curr_time + len_traj_pred + 1
    yaw = traj_data["yaw"][start_index:end_index]
    positions = traj_data["position"][start_index:end_index]
    goal_pos = traj_data["position"][goal_time]
    goal_yaw = traj_data["yaw"][goal_time]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred + 1,):
        print(f"WARNING: yaw shape mismatch! Expected {(len_traj_pred + 1,)}, got {yaw.shape}")

    waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
    waypoints_yaw = angle_difference(yaw[0], yaw)
    actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
    actions = actions[1:]
    
    goal_pos = to_local_coords(goal_pos.reshape(1, -1), positions[0], yaw[0]).squeeze(0)
    goal_yaw = angle_difference(yaw[0], goal_yaw)
    
    if True:  # normalize
        actions[:, :2] /= metric_waypoint_spacing
        goal_pos[:2] /= metric_waypoint_spacing
    
    goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
    return actions, goal_pos

# Test first 5 samples
for i in range(min(5, len(data))):
    print(f"\n=== Testing sample {i} ===")
    f_curr, curr_time, _, _ = data[i]
    print(f"Traj: {f_curr}, curr_time: {curr_time}")
    
    # Load trajectory data directly
    traj_path = os.path.join(data_folder, f_curr, "traj_data.pkl")
    with open(traj_path, "rb") as f:
        raw_traj_data = pickle.load(f)
    
    print(f"  Raw traj_data keys: {list(raw_traj_data.keys())}")
    for k, v in raw_traj_data.items():
        arr = np.asarray(v)
        print(f"    {k}: dtype={arr.dtype}, shape={arr.shape}, sample={arr[0] if len(arr) > 0 else 'empty'}")
    
    # Test _get_trajectory conversion
    converted = _get_trajectory(f_curr)
    for k, v in converted.items():
        print(f"    {k} (after _get_trajectory): dtype={v.dtype}, shape={v.shape}, sample={v[0] if len(v) > 0 else 'empty'}")
        if v.dtype == object:
            print(f"      WARNING: still object dtype! First elem type: {type(v[0])}")

print("\n=== Testing full pipeline (_compute_actions) ===")
f_curr, curr_time, goal_time, _ = data[0]
converted = _get_trajectory(f_curr)
actions, goal_pos = _compute_actions(converted, curr_time, goal_time)
if np.isnan(actions).any() or np.isinf(actions).any():
    print(f"  WARNING: actions has nan/inf! nan={np.isnan(actions).sum()}, inf={np.isinf(actions).sum()}")
if np.isnan(goal_pos).any() or np.isinf(goal_pos).any():
    print(f"  WARNING: goal_pos has nan/inf! nan={np.isnan(goal_pos).sum()}, inf={np.isinf(goal_pos).sum()}")
print(f"actions sample: {actions[0]}")
print(f"goal_pos sample: {goal_pos[0]}")

print("\n=== All tests completed ===")
