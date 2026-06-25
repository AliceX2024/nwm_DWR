#!/usr/bin/env python3
"""Debug script to check data structure differences between datasets."""

import pickle
import numpy as np
import os

def check_traj_data(data_folder, traj_name, dataset_name):
    """Check the structure of a trajectory data file."""
    # Different datasets might have different folder structures
    # Try common patterns
    possible_paths = [
        f"{data_folder}/{traj_name}/traj_data.pkl",
        f"{data_folder}/{traj_name}/instructor/0/traj_data.pkl",
        f"{data_folder}/{traj_name}/0/traj_data.pkl",
    ]
    
    traj_path = None
    for path in possible_paths:
        if os.path.exists(path):
            traj_path = path
            break
    
    if traj_path is None:
        print(f"[{dataset_name}] Could not find traj_data.pkl for {traj_name}")
        print(f"  Tried: {possible_paths}")
        # List what's in the traj folder
        traj_base = f"{data_folder}/{traj_name}"
        if os.path.exists(traj_base):
            print(f"  Contents of {traj_base}:")
            for item in os.listdir(traj_base)[:10]:
                print(f"    {item}/")
                sub_path = os.path.join(traj_base, item)
                if os.path.isdir(sub_path):
                    for subitem in os.listdir(sub_path)[:5]:
                        print(f"      {subitem}")
        return None
    
    print(f"[{dataset_name}] Loading: {traj_path}")
    
    with open(traj_path, 'rb') as f:
        traj = pickle.load(f)
    
    print(f"[{dataset_name}] Keys: {list(traj.keys())}")
    
    for k, v in traj.items():
        v_arr = np.asarray(v)
        print(f"  {k}:")
        print(f"    raw type: {type(v)}")
        print(f"    array shape: {v_arr.shape}")
        print(f"    array dtype: {v_arr.dtype}")
        
        if k == 'yaw':
            print(f"    yaw[0]: {v_arr[0]} (type: {type(v_arr[0])})")
            print(f"    yaw[:5]: {v_arr[:5]}")
            # Check if any element is array-like
            for i, val in enumerate(v_arr[:5]):
                if isinstance(val, (np.ndarray, list)):
                    print(f"      yaw[{i}] is array/list: {val}")
        if k == 'position':
            print(f"    position[:3]: {v_arr[:3]}")
    
    return traj

def main():
    print("=" * 60)
    print("Checking go_stanford dataset")
    print("=" * 60)
    
    # go_stanford
    traj_names_file = 'data_splits/go_stanford/test/traj_names.txt'
    with open(traj_names_file, 'r') as f:
        traj_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Total go_stanford trajs: {len(traj_names)}")
    
    # Check first traj
    traj_name = traj_names[0]
    print(f"\nFirst traj: {traj_name}")
    check_traj_data('data/go_stanford', traj_name, "go_stanford")
    
    # Check if the error happens at a specific index
    print("\n" + "=" * 60)
    print("Checking multiple trajectories for consistency")
    print("=" * 60)
    
    error_count = 0
    for i, traj_name in enumerate(traj_names[:20]):  # Check first 20
        try:
            traj = check_traj_data('data/go_stanford', traj_name, f"go_stanford[{i}]")
            if traj is not None:
                yaw = np.asarray(traj['yaw'])
                if yaw.ndim != 1:
                    print(f"  WARNING: yaw is not 1D! shape={yaw.shape}")
                    error_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            error_count += 1
    
    print(f"\nErrors found: {error_count}")
    
    # Compare with another dataset
    print("\n" + "=" * 60)
    print("Comparing with tartan_drive dataset")
    print("=" * 60)
    
    tartan_traj_names_file = 'data_splits/tartan_drive/test/traj_names.txt'
    if os.path.exists(tartan_traj_names_file):
        with open(tartan_traj_names_file, 'r') as f:
            tartan_traj_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Total tartan_drive trajs: {len(tartan_traj_names)}")
        if tartan_traj_names:
            check_traj_data('data/tartan_drive', tartan_traj_names[0], "tartan_drive")

if __name__ == "__main__":
    main()
