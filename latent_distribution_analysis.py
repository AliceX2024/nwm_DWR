#!/usr/bin/env python3
"""
Latent Space Distribution Analysis Script

Quantifies distribution shift between training and test sets using the VAE encoder
(Same VAE used during training: stabilityai/sd-vae-ft-ema).

Pipeline:
    frame/sequence
        ↓
    VAE encoder (encode → latent_dist.sample() → mul(0.18215))
        ↓
    latent (4, 28, 28) per frame
        ↓
    statistics (per-sequence pooling: mean over temporal+spatial dims → (4,) per frame)

Metrics:
  1. FID (Fréchet Inception Distance) — on per-frame (4,) pooled latents
  2. 2-Wasserstein distance (Gaussian) — between fitted Gaussians
  3. Covariance discrepancy (spectral norm of Σ_train^{-1/2}(Σ_test - Σ_train)Σ_train^{-1/2})
  4. Mean shift (||μ_train - μ_test||_2)

Types:
  Type I  — Zero-shot domain generalization: train vs go_stanford (unseen domain)
  Type II — Intra-domain OOD: train vs each of {recon, sacson, tartan_drive, scand} test splits
            (domain seen, specific frames unseen)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import torchvision.transforms.functional as TF
from torchvision import models, transforms as tv_transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# Data config (must match train.py / eval_config.yaml)
DATA_CONFIG = {
    "datasets": {
        "recon": {
            "data_folder": "data/recon",
            "train": "data_splits/recon/train/",
            "test":  "data_splits/recon/test/",
        },
        "sacson": {
            "data_folder": "data/sacson",
            "train": "my_data_splits/sacson/train",
            "test":  "my_data_splits/sacson/test",
        },
        "tartan_drive": {
            "data_folder": "data/tartan_drive",
            "train": "my_data_splits/tartan_drive/train/",
            "test":  "my_data_splits/tartan_drive/test/",
        },
        "scand": {
            "data_folder": "data/scand",
            "train": "my_data_splits/scand/train",
            "test":  "my_data_splits/scand/test",
        },
    },
    "eval_datasets": {
        "go_stanford": {
            "data_folder": "data/go_stanford",
            "test": "data_splits/go_stanford/test/",
        },
    },
    "image_size": 224,
    "context_size": 4,
}

# ---------------------------------------------------------------------------
# Image transform — identical to misc.transform
# ---------------------------------------------------------------------------
class CenterCropAR:
    def __call__(self, img):
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f"Invalid image size: ({w}, {h})")
        if w >= h:
            new_w = int(round(h * 4.0 / 3.0))
            left = (w - new_w) // 2
            img = TF.crop(img, top=0, left=left, height=h, width=new_w)
        else:
            new_h = int(round(w * 3.0 / 4.0))
            top = (h - new_h) // 2
            img = TF.crop(img, top=top, left=0, height=new_h, width=w)
        return img

transform = tv_transforms.Compose([
    CenterCropAR(),
    tv_transforms.Resize((224, 224)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_data_path(data_folder, f, time):
    return os.path.join(data_folder, f, f"{time}.jpg")


def load_image(data_folder, f, time):
    """Load and transform a single image. Returns tensor [3, 224, 224] in [-1,1]."""
    path = get_data_path(data_folder, f, time)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise IOError(f"Cannot load image {path}: {e}")
    return transform(img)


def build_pkl_path(split_folder, min_dist_cat, max_dist_cat, context_size, len_traj_pred):
    return os.path.join(
        split_folder,
        f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_n{context_size}_len_traj_pred_{len_traj_pred}.pkl",
    )


def load_index(pkl_path, min_dist_cat, max_dist_cat, context_size, len_traj_pred):
    """Load or rebuild the dataset index."""
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            return data[0]  # (index_to_data, goals_index)
        return data
    else:
        print(f"  [WARN] PKL not found: {pkl_path}, building index...")
        # For go_stanford, we need to build it ourselves
        return None


# ---------------------------------------------------------------------------
# Latent extraction helpers
# ---------------------------------------------------------------------------

def encode_images_to_latents(vae, images, device, dtype=torch.bfloat16):
    """
    Encode a batch of [N, 3, 224, 224] float tensors (in [-1,1])
    into latent tensors [N, 4, 28, 28].
    Matches the encoding used in train.py / isolated_nwm_infer.py:
        latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
    """
    images = images.to(device, dtype=torch.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        x = vae.encode(images).latent_dist.sample().mul_(0.18215)
    return x.float()


def pool_latents(x, pool_type="mean"):
    """
    Pool latent tensor [N, 4, 28, 28] → [N, 4] or [N, 16] or [N, 3136].
    pool_type: 'mean' → [N, 4]  (global avg over H,W)
               'adaptive_2x2' → [N, 16] (adaptive avg pool to 2x2)
               'adaptive_4x4' → [N, 64] (adaptive avg pool to 4x4)
               'flatten' → [N, 3136] (all channels + spatial)
    """
    if pool_type == "flatten":
        return x.flatten(start_dim=1)
    elif pool_type == "adaptive_2x2":
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(start_dim=1)
    elif pool_type == "adaptive_4x4":
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).flatten(start_dim=1)
    else:  # mean
        return x.mean(dim=[2, 3])


def extract_latents_for_split(vae, data_folder, split_folder, dataset_name,
                               min_dist_cat=-64, max_dist_cat=64,
                               context_size=4, len_traj_pred=64,
                               max_samples=None, pool_type="mean",
                               batch_size=64, device="cuda",
                               use_go_stanford_style=False):
    """
    Load images from a dataset split and encode them to latents.
    Returns an ndarray of pooled latents [N, D].

    If use_go_stanford_style=True, builds index from trajectory data files
    (for go_stanford which has no training-style PKL).
    """
    if not use_go_stanford_style:
        pkl_path = build_pkl_path(split_folder, min_dist_cat, max_dist_cat, context_size, len_traj_pred)
        print(f"  Loading index from: {pkl_path}")
        index_data = load_index(pkl_path, min_dist_cat, max_dist_cat, context_size, len_traj_pred)
        if index_data is None:
            return None
    else:
        # Build index directly from trajectory data files
        traj_names_file = os.path.join(split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            traj_names = [l.strip() for l in f.read().splitlines() if l.strip()]

        index_data = []
        skipped = 0
        for traj_name in tqdm(traj_names, desc=f"  Building index {dataset_name}", leave=False):
            traj_pkl = os.path.join(data_folder, traj_name, "traj_data.pkl")
            if not os.path.exists(traj_pkl):
                skipped += 1
                continue
            try:
                with open(traj_pkl, "rb") as f:
                    data = np.load(traj_pkl, allow_pickle=True)
                traj_len = len(data["position"])
            except Exception:
                skipped += 1
                continue
            if traj_len < context_size + len_traj_pred:
                continue
            for curr_time in range(context_size - 1, traj_len - len_traj_pred, 16):
                index_data.append((traj_name, curr_time, 0, 0))
        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} trajectories during index build")
        print(f"  Built index: {len(index_data)} samples for {dataset_name}/{split_folder}")

    # Load traj_names
    traj_names_file = os.path.join(split_folder, "traj_names.txt")
    with open(traj_names_file, "r") as f:
        traj_names = [l.strip() for l in f.read().splitlines() if l.strip()]

    # Build traj_name → set of used times from index_data
    used_times = {}
    for entry in index_data:
        f_curr = entry[0]
        curr_time = int(entry[1])
        if f_curr not in used_times:
            used_times[f_curr] = set()
        # Add context times
        for t in range(curr_time - context_size + 1, curr_time + 1):
            used_times[f_curr].add(t)

    # Collect unique (traj, time) pairs
    unique_pairs = []
    for traj, times in used_times.items():
        unique_pairs.extend([(traj, t) for t in sorted(times)])
    if max_samples:
        unique_pairs = unique_pairs[:max_samples]

    print(f"  {dataset_name}/{split_folder}: {len(index_data)} samples, {len(unique_pairs)} unique frames to encode")

    latents = []
    for i in tqdm(range(0, len(unique_pairs), batch_size), desc=f"  Encoding {dataset_name}", leave=False):
        batch_pairs = unique_pairs[i:i + batch_size]
        images = []
        for traj, t in batch_pairs:
            try:
                img = load_image(data_folder, traj, t)
                images.append(img)
            except Exception:
                continue
        if not images:
            continue
        imgs_tensor = torch.stack(images).to(device)
        lat = encode_images_to_latents(vae, imgs_tensor, device)
        pooled = pool_latents(lat, pool_type=pool_type)
        latents.append(pooled.cpu().numpy())

    if not latents:
        return None
    return np.concatenate(latents, axis=0)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def compute_fid(real_features, gen_features, eps=1e-6):
    """
    Compute FID between two sets of features.
    real_features: [N1, D]
    gen_features:  [N2, D]

    Uses the standard FID formula:
    FID = ||μ1 - μ2||^2 + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))
    """
    if real_features.shape[1] > 2000:
        # For very high-dim, use a subsampled approach to avoid covariance singular matrices
        rng = np.random.default_rng(42)
        idx = rng.choice(real_features.shape[1], size=min(2000, real_features.shape[1]), replace=False)
        real_features = real_features[:, idx]
        gen_features = gen_features[:, idx]

    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)

    diff = mu1 - mu2
    covmean = _sqrtm_safe(sigma1 @ sigma2, eps=eps)
    fid = np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def _sqrtm_safe(M, eps=1e-6):
    """Stable matrix square root via SVD."""
    M = (M + M.T) / 2  # symmetrize
    try:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, eps)
        return V @ np.diag(np.sqrt(w)) @ V.T
    except np.linalg.LinAlgError:
        # Fallback: regularized
        M = M + eps * np.eye(M.shape[0])
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, eps)
        return V @ np.diag(np.sqrt(w)) @ V.T


def compute_wasserstein_gaussian(real_features, gen_features):
    """
    2-Wasserstein distance between two Gaussian distributions fitted to the data.
    W2^2 = ||μ1 - μ2||^2 + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))
    """
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)

    mean_term = np.sum((mu1 - mu2) ** 2)
    covmean = _sqrtm_safe(sigma1 @ sigma2)
    cov_term = np.trace(sigma1 + sigma2 - 2 * covmean)
    w2_sq = mean_term + cov_term
    return float(np.sqrt(max(w2_sq, 0)))


def compute_cov_discrepancy(real_features, gen_features, eps=1e-6):
    """
    Spectral norm of Σ_train^{-1/2} (Σ_test - Σ_train) Σ_train^{-1/2}.
    Measures how differently the features are correlated.
    """
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)

    # Regularize
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    sigma1_inv_sqrt = _sqrtm_safe(np.linalg.inv(sigma1))
    diff = sigma1_inv_sqrt @ (sigma2 - sigma1) @ sigma1_inv_sqrt

    try:
        spectral_norm = float(np.max(np.abs(np.linalg.eigvalsh(diff))))
    except np.linalg.LinAlgError:
        spectral_norm = float(np.linalg.norm(diff, ord=2))
    return spectral_norm


def compute_mean_shift(real_features, gen_features):
    """Euclidean distance between feature means."""
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    return float(np.linalg.norm(mu1 - mu2))


# ---------------------------------------------------------------------------
# InceptionV3-based pixel-space FID
# ---------------------------------------------------------------------------

class InceptionV3Feature(nn.Module):
    """
    InceptionV3 feature extractor (pretrained on ImageNet).
    Returns features from the last pooling layer (before classifier, 2048-dim).
    """
    def __init__(self, device):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception = inception
        # Replace aux classifier to avoid side effects
        self.inception.aux_logits = False
        self.inception.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: [N, 3, 224, 224] in [0, 1] (ImageNet-normalized internally)
        Returns: [N, 2048]
        """
        # InceptionV3 expects inputs normalized with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = self.inception.avgpool(x)
        x = torch.flatten(x, 1)  # [N, 2048]
        return x


def extract_inception_features_for_split(data_folder, split_folder, dataset_name,
                                        min_dist_cat=-64, max_dist_cat=64,
                                        context_size=4, len_traj_pred=64,
                                        max_samples=None, batch_size=32,
                                        device="cuda", use_go_stanford_style=False):
    """
    Load images and extract InceptionV3 features.
    Returns an ndarray of features [N, 2048].
    If use_go_stanford_style=True, builds an index directly from traj_data.pkl files.
    """
    if not use_go_stanford_style:
        pkl_path = build_pkl_path(split_folder, min_dist_cat, max_dist_cat, context_size, len_traj_pred)
        print(f"  Loading index from: {pkl_path}")
        index_data = load_index(pkl_path, min_dist_cat, max_dist_cat, context_size, len_traj_pred)
        if index_data is None:
            print(f"  [SKIP] {dataset_name}/{split_folder}: no index")
            return None
    else:
        # Build index from go_stanford-style trajectories
        traj_names_file = os.path.join(split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            traj_names = [l.strip() for l in f.read().splitlines() if l.strip()]

        index_data = []
        skipped = 0
        for traj_name in tqdm(traj_names, desc=f"  Building index {dataset_name}", leave=False):
            traj_pkl = os.path.join(data_folder, traj_name, "traj_data.pkl")
            if not os.path.exists(traj_pkl):
                skipped += 1
                continue
            try:
                with open(traj_pkl, "rb") as f:
                    import numpy as _np
                    data = _np.load(traj_pkl, allow_pickle=True)
                traj_len = len(data["position"])
            except Exception:
                skipped += 1
                continue
            if traj_len < context_size + len_traj_pred:
                continue
            for curr_time in range(context_size - 1, traj_len - len_traj_pred, 16):
                index_data.append((traj_name, curr_time, 0, 0))

        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} trajectories during index build")
        print(f"  Built index: {len(index_data)} samples for {dataset_name}/{split_folder}")

    # Build used_times map
    used_times = {}
    for entry in index_data:
        f_curr = entry[0]
        curr_time = int(entry[1])
        if f_curr not in used_times:
            used_times[f_curr] = set()
        for t in range(curr_time - context_size + 1, curr_time + 1):
            used_times[f_curr].add(t)

    unique_pairs = []
    for traj, times in used_times.items():
        unique_pairs.extend([(traj, t) for t in sorted(times)])
    if max_samples:
        unique_pairs = unique_pairs[:max_samples]

    print(f"  {dataset_name}/{split_folder}: {len(index_data)} samples, {len(unique_pairs)} unique frames to extract Inception features")

    # We need raw [0,1] images for InceptionV3
    inception_transform = tv_transforms.Compose([
        CenterCropAR(),
        tv_transforms.Resize((299, 299)),  # InceptionV3 native resolution
        tv_transforms.ToTensor(),
    ])

    features = []
    for i in tqdm(range(0, len(unique_pairs), batch_size), desc=f"  Inception {dataset_name}", leave=False):
        batch_pairs = unique_pairs[i:i + batch_size]
        images = []
        for traj, t in batch_pairs:
            try:
                img = Image.open(get_data_path(data_folder, traj, t)).convert("RGB")
                img = inception_transform(img)
                images.append(img)
            except Exception:
                continue
        if not images:
            continue
        imgs = torch.stack(images).to(device)
        with torch.no_grad():
            feat = inception_v3_model(imgs)
        features.append(feat.cpu().numpy())

    if not features:
        return None
    return np.concatenate(features, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

inception_v3_model = None  # Lazy-loaded global

def main():
    global inception_v3_model

    parser = argparse.ArgumentParser(description="Latent Space Distribution Analysis")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max frames per split to encode (default: all)")
    parser.add_argument("--pool-type", type=str, default="mean",
                        choices=["mean", "adaptive_2x2", "adaptive_4x4", "flatten"],
                        help="Latent pooling strategy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="latent_distribution_results.npz")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Path to write markdown report (e.g. eval_results/latent_distribution_report.md)")
    parser.add_argument("--skip-inception", action="store_true",
                        help="Skip InceptionV3 pixel-space FID (faster, no ImageNet pretrained model download)")
    args = parser.parse_args()

    if args.max_samples == 0:
        args.max_samples = None  # 0 means "use all"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load VAE
    # -------------------------------------------------------------------------
    print("\n[1] Loading VAE...")
    vae_model_name = os.getenv("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-ema")
    try:
        vae = AutoencoderKL.from_pretrained(vae_model_name, local_files_only=True).to(device)
    except Exception:
        print(f"  Local VAE not found, downloading from Hugging Face...")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
    vae.eval()
    print(f"  VAE loaded: {vae_model_name}")

    # -------------------------------------------------------------------------
    # 2. Load InceptionV3 (for pixel-space FID)
    # -------------------------------------------------------------------------
    if not args.skip_inception:
        print("\n[2] Loading InceptionV3 (pixel-space FID)...")
        inception_v3_model = InceptionV3Feature(device)
        inception_v3_model.to(device)
        print("  InceptionV3 loaded.")
    else:
        inception_v3_model = None
        print("\n[2] Skipping InceptionV3 (--skip-inception set).")

    # -------------------------------------------------------------------------
    # 3. Extract VAE latents for training set
    # -------------------------------------------------------------------------
    print("\n[3] Encoding training set (recon + sacson + tartan_drive + scand)...")
    train_latents = []
    for ds_name, ds_cfg in DATA_CONFIG["datasets"].items():
        lat = extract_latents_for_split(
            vae, ds_cfg["data_folder"], ds_cfg["train"], ds_name,
            context_size=DATA_CONFIG["context_size"],
            max_samples=args.max_samples,
            pool_type=args.pool_type,
            batch_size=args.batch_size,
            device=device,
        )
        if lat is not None:
            print(f"  → {ds_name}/train: {lat.shape[0]} frames encoded")
            train_latents.append(lat)
        else:
            print(f"  → {ds_name}/train: SKIPPED")

    train_latents = np.concatenate(train_latents, axis=0)
    print(f"  TOTAL train VAE latents: {train_latents.shape}")

    # -------------------------------------------------------------------------
    # 4. Extract Inception features for training set
    # -------------------------------------------------------------------------
    train_inception = None
    if inception_v3_model is not None:
        print("\n[4] Extracting InceptionV3 features for training set...")
        train_inception_parts = []
        for ds_name, ds_cfg in DATA_CONFIG["datasets"].items():
            feat = extract_inception_features_for_split(
                ds_cfg["data_folder"], ds_cfg["train"], ds_name,
                context_size=DATA_CONFIG["context_size"],
                max_samples=args.max_samples,
                batch_size=32,
                device=device,
            )
            if feat is not None:
                print(f"  → {ds_name}/train: {feat.shape[0]} frames extracted")
                train_inception_parts.append(feat)
            else:
                print(f"  → {ds_name}/train: SKIPPED")
        if train_inception_parts:
            train_inception = np.concatenate(train_inception_parts, axis=0)
            print(f"  TOTAL train Inception features: {train_inception.shape}")

    # -------------------------------------------------------------------------
    # 5. Extract latents for Type II test sets
    # -------------------------------------------------------------------------
    print("\n[5] Encoding Type II test sets...")
    type2_latents = {}
    for ds_name, ds_cfg in DATA_CONFIG["datasets"].items():
        lat = extract_latents_for_split(
            vae, ds_cfg["data_folder"], ds_cfg["test"], ds_name,
            context_size=DATA_CONFIG["context_size"],
            max_samples=args.max_samples,
            pool_type=args.pool_type,
            batch_size=args.batch_size,
            device=device,
        )
        if lat is not None:
            print(f"  → {ds_name}/test: {lat.shape[0]} frames encoded")
            type2_latents[ds_name] = lat
        else:
            print(f"  → {ds_name}/test: SKIPPED")

    # -------------------------------------------------------------------------
    # 6. Extract Inception features for Type II test sets
    # -------------------------------------------------------------------------
    type2_inception = {}
    if inception_v3_model is not None:
        print("\n[6] Extracting InceptionV3 features for Type II test sets...")
        for ds_name, ds_cfg in DATA_CONFIG["datasets"].items():
            feat = extract_inception_features_for_split(
                ds_cfg["data_folder"], ds_cfg["test"], ds_name,
                context_size=DATA_CONFIG["context_size"],
                max_samples=args.max_samples,
                batch_size=32,
                device=device,
            )
            if feat is not None:
                print(f"  → {ds_name}/test: {feat.shape[0]} frames extracted")
                type2_inception[ds_name] = feat
            else:
                print(f"  → {ds_name}/test: SKIPPED")

    # -------------------------------------------------------------------------
    # 7. Extract latents for Type I test set (go_stanford)
    # -------------------------------------------------------------------------
    print("\n[7] Encoding Type I test set (go_stanford)...")
    go_stanford_latents = None
    for ds_name, ds_cfg in DATA_CONFIG["eval_datasets"].items():
        if ds_name == "go_stanford":
            # go_stanford has no training-style PKL → build index from trajectories
            lat = extract_latents_for_split(
                vae, ds_cfg["data_folder"], ds_cfg["test"], ds_name,
                context_size=DATA_CONFIG["context_size"],
                max_samples=args.max_samples,
                pool_type=args.pool_type,
                batch_size=args.batch_size,
                device=device,
                use_go_stanford_style=True,
            )
            if lat is not None:
                print(f"  → go_stanford/test: {lat.shape[0]} frames encoded")
                go_stanford_latents = lat
            else:
                print(f"  → go_stanford/test: SKIPPED")

    # -------------------------------------------------------------------------
    # 8. Extract Inception features for Type I test set
    # -------------------------------------------------------------------------
    go_stanford_inception = None
    if inception_v3_model is not None:
        print("\n[8] Extracting InceptionV3 features for Type I (go_stanford)...")
        for ds_name, ds_cfg in DATA_CONFIG["eval_datasets"].items():
            if ds_name == "go_stanford":
                feat = extract_inception_features_for_split(
                    ds_cfg["data_folder"], ds_cfg["test"], ds_name,
                    context_size=DATA_CONFIG["context_size"],
                    max_samples=args.max_samples,
                    batch_size=32,
                    device=device,
                    use_go_stanford_style=True,
                )
                if feat is not None:
                    print(f"  → go_stanford/test: {feat.shape[0]} frames extracted")
                    go_stanford_inception = feat
                else:
                    print(f"  → go_stanford/test: SKIPPED")

    # -------------------------------------------------------------------------
    # 9. Compute metrics
    # -------------------------------------------------------------------------
    print("\n[9] Computing distribution metrics...")
    print(f"  Pool type: {args.pool_type}, VAE latent dim: {train_latents.shape[1]}")
    print()

    def _print_header():
        print(f"{'Dataset':<20} {'Type':>4}  {'FID(VAE)':>10}  {'FID(Incep)':>10}  {'W2':>8}  {'MeanShift':>10}  {'CovDisc':>8}")
        print("  " + "-" * 82)

    _print_header()

    all_results = {}

    for ds_name in ["recon", "sacson", "tartan_drive", "scand"]:
        if ds_name not in type2_latents:
            continue
        test_lat = type2_latents[ds_name]
        fid_vae = compute_fid(train_latents, test_lat)
        w2  = compute_wasserstein_gaussian(train_latents, test_lat)
        ms  = compute_mean_shift(train_latents, test_lat)
        cd  = compute_cov_discrepancy(train_latents, test_lat)

        fid_inception_val = None
        fid_incep_str = "     N/A"
        if inception_v3_model is not None and ds_name in type2_inception:
            fid_inception_val = compute_fid(train_inception, type2_inception[ds_name])
            fid_incep_str = f"{fid_inception_val:10.4f}"

        print(f"{ds_name}_test{' ':>{20-len(ds_name)-5}}  II  {fid_vae:10.4f}  {fid_incep_str:>10}  {w2:8.4f}  {ms:10.4f}  {cd:8.4f}")
        all_results[ds_name] = {
            "type": "II",
            "fid_vae": fid_vae,
            "fid_inception": fid_inception_val,
            "w2": w2, "mean_shift": ms, "cov_disc": cd,
        }

    # Type I: go_stanford
    if go_stanford_latents is not None:
        fid_vae = compute_fid(train_latents, go_stanford_latents)
        w2  = compute_wasserstein_gaussian(train_latents, go_stanford_latents)
        ms  = compute_mean_shift(train_latents, go_stanford_latents)
        cd  = compute_cov_discrepancy(train_latents, go_stanford_latents)

        fid_inception_val = None
        fid_incep_str = "     N/A"
        if inception_v3_model is not None and go_stanford_inception is not None:
            fid_inception_val = compute_fid(train_inception, go_stanford_inception)
            fid_incep_str = f"{fid_inception_val:10.4f}"

        print(f"{'go_stanford':<20}     I  {fid_vae:10.4f}  {fid_incep_str:>10}  {w2:8.4f}  {ms:10.4f}  {cd:8.4f}")
        all_results["go_stanford"] = {
            "type": "I",
            "fid_vae": fid_vae,
            "fid_inception": fid_inception_val,
            "w2": w2, "mean_shift": ms, "cov_disc": cd,
        }

    print()

    # -------------------------------------------------------------------------
    # 10. Summary tables
    # -------------------------------------------------------------------------
    print("\n[10] Summary — VAE FID (latent-space, pooled):")
    fid_table = [(k, v["fid_vae"], v["type"]) for k, v in all_results.items()]
    fid_table.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, fid, typ) in enumerate(fid_table, 1):
        marker = " ★ (Type I — unseen domain)" if typ == "I" else "   (Type II — intra-domain)"
        print(f"  {rank}. {name}: FID={fid:.4f}{marker}")

    if inception_v3_model is not None:
        print("\n[11] Summary — InceptionV3 FID (pixel-space, 2048-dim):")
        incep_results = [(k, v["fid_inception"], v["type"])
                         for k, v in all_results.items() if v["fid_inception"] is not None]
        incep_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, fid, typ) in enumerate(incep_results, 1):
            marker = " ★ (Type I — unseen domain)" if typ == "I" else "   (Type II — intra-domain)"
            print(f"  {rank}. {name}: FID={fid:.4f}{marker}")

    # -------------------------------------------------------------------------
    # 12. Per-dataset statistics
    # -------------------------------------------------------------------------
    print("\n[12] Per-dataset VAE latent statistics:")
    print(f"  {'Dataset':<25}  {'N':>8}  {'Mean(μ)':>12}  {'Std':>8}  {'Min':>10}  {'Max':>10}")
    print("  " + "-" * 80)
    for name, latents in list(type2_latents.items()) + ([("go_stanford", go_stanford_latents)] if go_stanford_latents is not None else []):
        lat = np.mean(latents, axis=0) if latents is not None else None
        if lat is None:
            continue
        print(f"  {name:<25}  {latents.shape[0]:>8}  {np.mean(lat):>12.6f}  {np.std(lat):>8.4f}  {np.min(lat):>10.4f}  {np.max(lat):>10.4f}")
    print(f"  {'TRAIN (combined)':<25}  {train_latents.shape[0]:>8}  {np.mean(np.mean(train_latents, axis=0)):>12.6f}  {np.std(train_latents):>8.4f}  {np.min(train_latents):>10.4f}  {np.max(train_latents):>10.4f}")

    # -------------------------------------------------------------------------
    # 13. Save results
    # -------------------------------------------------------------------------
    save_kwargs = {
        "train_latents": train_latents,
        **{f"{k}_latents": v for k, v in type2_latents.items()},
        **{f"{k}_inception": v for k, v in type2_inception.items()},
        "all_results": all_results,
    }
    if go_stanford_latents is not None:
        save_kwargs["go_stanford_latents"] = go_stanford_latents
    if go_stanford_inception is not None:
        save_kwargs["go_stanford_inception"] = go_stanford_inception
    if train_inception is not None:
        save_kwargs["train_inception"] = train_inception

    # -------------------------------------------------------------------------
    # 13. Write markdown report
    # -------------------------------------------------------------------------
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            pool_dim = train_latents.shape[1]
            f.write(f"# Latent Distribution Analysis Report\n\n")
            f.write(f"**Pool type:** `{args.pool_type}`  |  ")
            f.write(f"**VAE latent dim:** {pool_dim}  |  ")
            f.write(f"**Inception FID:** {'enabled' if inception_v3_model else 'skipped'}\n\n")

            # --- Table 1: Full metrics ---
            f.write("## Distribution Metrics\n\n")
            if inception_v3_model:
                header = "| Dataset | Type | VAE FID | Inception FID | W2 | MeanShift | CovDisc |"
                sep    = "|---------|:----:|--------:|--------------:|----:|----------:|--------:|"
            else:
                header = "| Dataset | Type | VAE FID | W2 | MeanShift | CovDisc |"
                sep    = "|---------|:----:|--------:|----:|----------:|--------:|"
            f.write(header + "\n" + sep + "\n")

            for ds_name, res in all_results.items():
                typ_marker = "I ★" if res["type"] == "I" else "II"
                if inception_v3_model and res["fid_inception"] is not None:
                    row = f"| {ds_name} | {typ_marker} | {res['fid_vae']:.4f} | {res['fid_inception']:.4f} | {res['w2']:.4f} | {res['mean_shift']:.4f} | {res['cov_disc']:.4f} |"
                elif inception_v3_model:
                    row = f"| {ds_name} | {typ_marker} | {res['fid_vae']:.4f} | N/A | {res['w2']:.4f} | {res['mean_shift']:.4f} | {res['cov_disc']:.4f} |"
                else:
                    row = f"| {ds_name} | {typ_marker} | {res['fid_vae']:.4f} | {res['w2']:.4f} | {res['mean_shift']:.4f} | {res['cov_disc']:.4f} |"
                f.write(row + "\n")
            f.write("\n")

            # --- Table 2: VAE latent statistics ---
            f.write("## Per-Dataset Latent Statistics\n\n")
            f.write(f"| Dataset | N | Mean(μ) | Std | Min | Max |\n")
            f.write(f"|---------|--:|--------:|----:|-----:|-----:|\n")
            for name, latents in list(type2_latents.items()) + ([("go_stanford", go_stanford_latents)] if go_stanford_latents is not None else []):
                if latents is None:
                    continue
                mu = np.mean(latents)
                std = np.std(latents)
                mn = np.min(latents)
                mx = np.max(latents)
                f.write(f"| {name} | {latents.shape[0]} | {mu:.6f} | {std:.4f} | {mn:.4f} | {mx:.4f} |\n")
            f.write(f"| **TRAIN (combined)** | **{train_latents.shape[0]}** | {np.mean(train_latents):.6f} | {np.std(train_latents):.4f} | {np.min(train_latents):.4f} | {np.max(train_latents):.4f} |\n")
            f.write("\n")

            # --- Interpretation ---
            f.write("## Interpretation\n\n")
            f.write("| Metric | Meaning |\n")
            f.write("|--------|--------|\n")
            f.write("| **VAE FID** | Distribution distance in VAE latent space (pooled to dim={pool_dim}). Lower = closer to training latent distribution. |\n")
            f.write("| **Inception FID** | Distribution distance in 2048-dim pixel-semantic space. Lower = pixel content more similar to training images. |\n")
            f.write("| **W2** | 2-Wasserstein distance between fitted Gaussians. Robust綜合衡量均值+协方差差异. |\n")
            f.write("| **MeanShift** | L2 distance between mean vectors. Pure mean offset. |\n")
            f.write("| **CovDisc** | Spectral norm of normalized covariance difference. Measures how differently channels are correlated vs training. |\n")
            f.write("\n")
            f.write(f"*Generated by latent_distribution_analysis.py | pool={args.pool_type} | samples={'all' if args.max_samples is None else args.max_samples}*\n")
        print(f"[Report] Written to: {args.output_md}")

    np.savez(args.output, **save_kwargs)
    print(f"\n[Done] Results saved to: {args.output}")


if __name__ == "__main__":
    main()

