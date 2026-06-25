# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
FVD (Fréchet Video Distance) evaluation script.

Computes FVD metric between ground truth and predicted videos using 
pretrained I3D network features.

Usage:
    python compute_fvd.py \
        --gt_dir /path/to/gt \
        --exp_dir /path/to/exp \
        --datasets tartan_drive \
        --eval_types rollout \
        --rollout_fps_values 1 \
        --num_frames 16
"""
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import json

from PIL import Image
from torchvision import transforms
import distributed as dist


def compute_fvd_from_features(real_features, fake_features):
    """Compute FVD from pre-extracted features.
    
    Args:
        real_features: numpy array of shape (N, feature_dim)
        fake_features: numpy array of shape (N, feature_dim)
    
    Returns:
        FVD score (lower is better)
    """
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu_real - mu_fake
    
    # Compute matrix square root of sigma_real @ sigma_fake
    covmean, _ = sqrt_matrix(sigma_real, sigma_fake)
    
    fvd = np.sqrt(np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean))
    return float(fvd)


def sqrt_matrix(A, B):
    """Compute matrix square root of A @ B using SVD."""
    if A.shape != B.shape:
        return np.zeros_like(A), None
    try:
        # Use SVD-based method for stability
        eps = 1e-7
        product = A @ B
        eigvals, eigvecs = np.linalg.eig(product)
        eigvals = np.sqrt(np.abs(eigvals)) + eps
        return np.dot(eigvecs * eigvals, eigvecs.T), True
    except Exception:
        return np.zeros_like(A), None


class I3DFeatureExtractor:
    """Extract features using pretrained I3D network."""
    
    def __init__(self, device):
        self.device = device
        self.i3d = None
        self._load_i3d()
    
    def _load_i3d(self):
        """Load pretrained I3D model."""
        try:
            from pytorch_fvd.fvd import load_i3d_pretrained
            self.i3d = load_i3d_pretrained().to(self.device)
            self.i3d.eval()
            print("Successfully loaded I3D from pytorch_fvd")
        except ImportError:
            try:
                # Alternative: use torchvideocnn if available
                import torchvideocnn
                self.i3d = torchvideocnn.I3D(400, pretrained=True).to(self.device)
                self.i3d.eval()
                print("Successfully loaded I3D from torchvideocnn")
            except ImportError:
                print("Warning: pytorch_fvd or torchvideocnn not available")
                self.i3d = None
    
    @torch.no_grad()
    def extract_features(self, videos):
        """Extract I3D features from videos.
        
        Args:
            videos: Tensor of shape (B, T, C, H, W) in [0, 1] range
        
        Returns:
            Features tensor of shape (B, feature_dim)
        """
        if self.i3d is None:
            raise RuntimeError("I3D model not loaded")
        
        import torch.nn.functional as F
        
        # Resize to 224x224
        videos = F.interpolate(videos, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1]
        videos = videos * 2 - 1
        # Permute to (B, C, T, H, W) for I3D
        videos = videos.permute(0, 2, 1, 3, 4)
        
        features = self.i3d(videos)
        return features


def load_video_frames(video_dir, num_frames, frame_indices=None):
    """Load frames from a video directory.
    
    Args:
        video_dir: Directory containing frame images (0.png, 1.png, ...)
        num_frames: Number of frames to load
        frame_indices: Optional specific frame indices to load
    
    Returns:
        Tensor of shape (num_frames, C, H, W) in [0, 1] range
    """
    frames = []
    
    if frame_indices is None:
        frame_indices = range(num_frames)
    
    for idx in frame_indices:
        frame_path = os.path.join(video_dir, f'{idx}.png')
        if os.path.exists(frame_path):
            img = transforms.ToTensor()(Image.open(frame_path).convert("RGB"))
            frames.append(img)
        else:
            # If frame doesn't exist, use zero tensor as placeholder
            if frames:
                frames.append(torch.zeros_like(frames[0]))
            else:
                raise FileNotFoundError(f"Frame {frame_path} not found")
    
    return torch.stack(frames, dim=0)


def evaluate_fvd(args, dataset_name, eval_type, gt_dir, exp_dir, rollout_fps=None, num_frames=16):
    """Evaluate FVD for a single dataset and eval type."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = I3DFeatureExtractor(device)
    
    if extractor.i3d is None:
        print("I3D not available, cannot compute FVD")
        return None
    
    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        total_frames = num_frames
        frame_indices = [i * rollout_fps for i in range(total_frames)]
    elif eval_type == 'time':
        eval_name = eval_type
        total_frames = num_frames
        frame_indices = list(range(total_frames))
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    
    eps = sorted(os.listdir(gt_dir))
    
    all_real_features = []
    all_fake_features = []
    
    for ep in tqdm(eps, desc=f'Extracting FVD features for {dataset_name}_{eval_name}'):
        gt_ep_dir = os.path.join(gt_dir, ep)
        exp_ep_dir = os.path.join(exp_dir, ep)
        
        if not os.path.isdir(gt_ep_dir) or not os.path.isdir(exp_ep_dir):
            continue
        
        try:
            # Load video frames
            real_frames = load_video_frames(gt_ep_dir, total_frames, frame_indices).unsqueeze(0)
            fake_frames = load_video_frames(exp_ep_dir, total_frames, frame_indices).unsqueeze(0)
            
            # Extract features
            real_feats = extractor.extract_features(real_frames.to(device))
            fake_feats = extractor.extract_features(fake_frames.to(device))
            
            all_real_features.append(real_feats.cpu().numpy())
            all_fake_features.append(fake_feats.cpu().numpy())
            
        except Exception as e:
            print(f"Error processing {ep}: {e}")
            continue
    
    if not all_real_features:
        print(f"No valid videos found for {dataset_name}_{eval_name}")
        return None
    
    # Concatenate all features
    real_features = np.concatenate(all_real_features, axis=0)
    fake_features = np.concatenate(all_fake_features, axis=0)
    
    print(f"Real features shape: {real_features.shape}")
    print(f"Fake features shape: {fake_features.shape}")
    
    # Compute FVD
    fvd_score = compute_fvd_from_features(real_features, fake_features)
    print(f"FVD for {dataset_name}_{eval_name}: {fvd_score:.4f}")
    
    return fvd_score


def main(args):
    device = 'cuda'
          
    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    results = {}
    
    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                try:
                    print(f"\nEvaluating FVD: rollout {rollout_fps}fps, {dataset_name}")
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    
                    fvd_score = evaluate_fvd(
                        args, dataset_name, 'rollout', 
                        gt_dataset_rollout_dir, exp_dataset_rollout_dir, 
                        rollout_fps, args.num_frames
                    )
                    
                    if fvd_score is not None:
                        results[f'{dataset_name}_{eval_name}_fvd'] = fvd_score
                        
                except Exception as e:
                    print(f"Error evaluating rollout {rollout_fps} for {dataset_name}: {e}")

        if 'time' in args.eval_types:
            try:
                print(f"\nEvaluating FVD: time, {dataset_name}")
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                
                fvd_score = evaluate_fvd(
                    args, dataset_name, 'time',
                    gt_dataset_time_dir, exp_dataset_time_dir,
                    None, args.num_frames
                )
                
                if fvd_score is not None:
                    results[f'{dataset_name}_{eval_name}_fvd'] = fvd_score
                    
            except Exception as e:
                print(f"Error evaluating time for {dataset_name}: {e}")
    
    # Save results
    if results:
        output_fn = os.path.join(args.exp_dir, 'fvd_results.json')
        with open(output_fn, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nFVD results saved to {output_fn}")
        print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute FVD metrics')
    
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_types", type=str, default='rollout', help="evaluations (time, rollout)")
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name (comma separated)")
    
    parser.add_argument("--input_fps", type=int, default=4, help="input fps")
    parser.add_argument("--rollout_fps_values", type=str, default='1', help="comma separated fps values")
    parser.add_argument("--num_frames", type=int, default=16, help="number of frames for FVD")
    
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    args.eval_types = args.eval_types.split(',')
    
    main(args)
