"""
Dataset Distribution Divergence Analysis
=========================================
Computes multiple quantitative metrics to measure distribution shift between
your 5 datasets: recon, tartan_drive, sacson, scand (seen) vs go_stanford (unseen).

Metrics:
  1. FID (Fréchet Inception Distance)    - Gaussian-assumption distance in Inception feature space
  2. Inception Score (IS)                - Per-dataset quality/diversity proxy
  3. Mean/Std image statistics           - Per-channel RGB means & stds, edge density
  4. Wasserstein-1 distance              - 1D histogram-based pixel-level shift
  5. Domain-Classifier AUC               - Binary classifier trained to distinguish dataset pairs
  6. KL Divergence                       - Per-channel KL on normalized histograms
"""

import os
import sys
import json
import warnings
import hashlib
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/DATA/DATANAS2/xiaoyj25/projects/nwm/data")
DATASETS  = ["recon", "tartan_drive", "sacson", "scand", "go_stanford"]
SAMPLES   = 500          # images per dataset for heavy computations
SEED      = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image loading helpers ───────────────────────────────────────────────────
def list_image_paths(root: Path, max_per_dataset: int = None) -> list[Path]:
    """Recursively collect all .jpg/.png images under root."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(Path(p) for p in glob(str(root / ext), recursive=True))
    if max_per_dataset and len(paths) > max_per_dataset:
        paths = list(np.random.choice(paths, max_per_dataset, replace=False))
    return sorted(paths)

def pil_loader(path: Path) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

# ── Pre-trained InceptionV3 for feature extraction ──────────────────────────
print("Loading InceptionV3 (pretrained) ...")
inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
inception.fc = nn.Identity()   # Remove classification head → 2048-dim features
inception = inception.to(device).eval()

transform_inception = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_inception_features(image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool features from a list of images."""
    features, batch = [], []
    with torch.no_grad():
        for p in image_paths:
            img = pil_loader(p)
            if img is None:
                continue
            batch.append(transform_inception(img))
            if len(batch) == batch_size:
                x = torch.stack(batch).to(device)
                f = inception(x).cpu().numpy()
                features.append(f)
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            f = inception(x).cpu().numpy()
            features.append(f)
    if not features:
        return np.zeros((1, 2048))
    return np.concatenate(features, axis=0)

# ── Low-level image statistics ───────────────────────────────────────────────
def compute_pixel_stats(image_paths: list[Path]) -> dict:
    """RGB mean/std, grayscale mean/std, edge density (Sobel magnitude)."""
    r_vals, g_vals, b_vals, gray_vals, edge_vals = [], [], [], [], []
    for p in image_paths:
        img = pil_loader(p)
        if img is None:
            continue
        arr = np.array(img, dtype=np.float32) / 255.0
        r_vals.append(arr[:,:,0].ravel())
        g_vals.append(arr[:,:,1].ravel())
        b_vals.append(arr[:,:,2].ravel())
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        gray_vals.append(gray.ravel())
        # Sobel edge magnitude
        from scipy.ndimage import sobel
        ex = sobel(gray, axis=0)
        ey = sobel(gray, axis=1)
        edge_vals.append(np.sqrt(ex**2 + ey**2).ravel())

    def stats1d(vals):
        all_vals = np.concatenate(vals)
        return float(np.mean(all_vals)), float(np.std(all_vals))

    r_mean, r_std = stats1d(r_vals)
    g_mean, g_std = stats1d(g_vals)
    b_mean, b_std = stats1d(b_vals)
    gray_mean, gray_std = stats1d(gray_vals)
    edge_mean, edge_std = stats1d(edge_vals)

    return {
        "r": {"mean": r_mean, "std": r_std},
        "g": {"mean": g_mean, "std": g_std},
        "b": {"mean": b_mean, "std": b_std},
        "gray": {"mean": gray_mean, "std": gray_std},
        "edge": {"mean": edge_mean, "std": edge_std},
    }

# ── Metric 1: FID ────────────────────────────────────────────────────────────
def compute_fid(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Fréchet Inception Distance between two sets of Inception features.
    Assumes features ~ N(mu1, sigma1) and N(mu2, sigma2).
    """
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    # Numerical stability
    diff = mu1 - mu2
    covmean = np.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return max(0.0, fid)

# ── Metric 2: Inception Score (approximation using features) ────────────────
def inception_score_approx(features: np.ndarray, eps: float = 1e-8) -> float:
    """
    Approximate IS via label-probability proxy.
    We use the InceptionV3 final-logits proxy: train a small classifier on
    ImageNet-style features to distinguish images within the dataset.
    Higher within-dataset variance → more diverse → higher score.
    Here: we return the mean pairwise cosine distance as a diversity proxy.
    """
    norms = features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)
    cosine_sim = norms @ norms.T
    triu_idx = np.triu_indices(cosine_sim.shape[0], k=1)
    return float(np.mean(1.0 - cosine_sim[triu_idx]))  # 0 = identical, 1 = maximally diverse

# ── Metric 3: Wasserstein-1 pixel distance ───────────────────────────────────
def compute_wasserstein_pixel(image_paths: list[Path], n_bins: int = 256) -> dict:
    """W1 distance per channel between two image sets (call pairwise later)."""
    channels = {"r": [], "g": [], "b": [], "gray": []}
    for p in image_paths:
        img = pil_loader(p)
        if img is None:
            continue
        arr = np.array(img, dtype=np.float32) / 255.0
        channels["r"].append(arr[:,:,0].ravel())
        channels["g"].append(arr[:,:,1].ravel())
        channels["b"].append(arr[:,:,2].ravel())
        channels["gray"].append(0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
    results = {}
    for ch, vals_list in channels.items():
        all_vals = np.concatenate(vals_list)
        hist, bin_edges = np.histogram(all_vals, bins=n_bins, range=(0.0, 1.0), density=True)
        hist = hist / (hist.sum() + eps)
        results[ch] = {"hist": hist, "edges": bin_edges}
    return results

eps = 1e-10

# ── Metric 4: KL Divergence per channel ────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

# ── Metric 5: Domain-Classifier AUC ────────────────────────────────────────
def compute_domain_classifier_auc(features1: np.ndarray, features2: np.ndarray) -> float:
    """Train a logistic regression to distinguish dataset A vs B. AUC = domain gap."""
    n1, n2 = len(features1), len(features2)
    X = np.vstack([features1, features2])
    y = np.array([0]*n1 + [1]*n2)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    # Quick split for AUC
    split = int(0.7 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, proba)
    except ValueError:
        auc = 0.5
    return float(auc)

# ── Collect images per dataset ──────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: Loading image paths (sampling up to {} per dataset) ...".format(SAMPLES))
print("="*70)
dataset_paths = {}
for ds in DATASETS:
    paths = list_image_paths(DATA_ROOT / ds)
    # Sample uniformly from all found
    if len(paths) > SAMPLES:
        paths = list(np.random.choice(paths, SAMPLES, replace=False))
    dataset_paths[ds] = paths
    print(f"  {ds:<15}: {len(paths)} images loaded")

# ── Extract Inception features ───────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: Extracting InceptionV3 features ...")
print("="*70)
dataset_features = {}
for ds in DATASETS:
    print(f"  Extracting {ds} ...", end=" ", flush=True)
    dataset_features[ds] = extract_inception_features(dataset_paths[ds])
    print(f"  → shape={dataset_features[ds].shape}")

# ── Compute per-dataset statistics ─────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: Computing pixel-level statistics ...")
print("="*70)
dataset_pixel_stats = {}
for ds in DATASETS:
    print(f"  Computing pixel stats for {ds} ...", end=" ", flush=True)
    dataset_pixel_stats[ds] = compute_pixel_stats(dataset_paths[ds])
    s = dataset_pixel_stats[ds]
    print(f"  RGB=({s['r']['mean']:.3f},{s['g']['mean']:.3f},{s['b']['mean']:.3f}), "
          f"Gray={s['gray']['mean']:.3f}±{s['gray']['std']:.3f}, "
          f"Edge={s['edge']['mean']:.3f}±{s['edge']['std']:.3f}")

# ── FID matrix ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: Computing FID matrix (Intra vs Inter domain) ...")
print("="*70)
fid_matrix = {}
for ds1 in DATASETS:
    for ds2 in DATASETS:
        fid_matrix[(ds1, ds2)] = compute_fid(dataset_features[ds1], dataset_features[ds2])
        if ds1 < ds2:
            print(f"  FID({ds1:<12}, {ds2:<12}) = {fid_matrix[(ds1, ds2)]:.4f}")

# ── Domain-Classifier AUC matrix ────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: Computing Domain-Classifier AUC ...")
print("="*70)
auc_matrix = {}
for ds1 in DATASETS:
    for ds2 in DATASETS:
        if ds1 >= ds2:
            continue
        auc = compute_domain_classifier_auc(dataset_features[ds1], dataset_features[ds2])
        auc_matrix[(ds1, ds2)] = auc
        print(f"  AUC({ds1:<12}, {ds2:<12}) = {auc:.4f}")

# ── Wasserstein-1 per channel ────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6: Computing Wasserstein-1 distance (pixel-level) ...")
print("="*70)
# Build pixel histograms first
dataset_histograms = {}
for ds in DATASETS:
    print(f"  Computing histograms for {ds} ...", end=" ", flush=True)
    dataset_histograms[ds] = compute_wasserstein_pixel(dataset_paths[ds])
    print("done")

w1_matrix = {}
for ds1 in DATASETS:
    for ds2 in DATASETS:
        if ds1 >= ds2:
            continue
        h1, h2 = dataset_histograms[ds1], dataset_histograms[ds2]
        w1_total = 0.0
        for ch in ["r", "g", "b", "gray"]:
            w1_total += wasserstein_distance(h1[ch]["hist"], h2[ch]["hist"])
        w1_matrix[(ds1, ds2)] = w1_total / 4
        print(f"  W1({ds1:<12}, {ds2:<12}) = {w1_matrix[(ds1, ds2)]:.6f}")

# ── Inception Score approximation ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 7: Computing Inception Score (diversity proxy) ...")
print("="*70)
is_scores = {}
for ds in DATASETS:
    score = inception_score_approx(dataset_features[ds])
    is_scores[ds] = score
    print(f"  IS({ds:<15}) = {score:.4f}")

# ── Per-channel KL divergence ─────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 8: Computing KL Divergence per channel ...")
print("="*70)
kl_matrix = {}
for ds1 in DATASETS:
    for ds2 in DATASETS:
        if ds1 >= ds2:
            continue
        h1, h2 = dataset_histograms[ds1], dataset_histograms[ds2]
        kl_total = 0.0
        for ch in ["r", "g", "b", "gray"]:
            kl_total += kl_divergence(h1[ch]["hist"], h2[ch]["hist"])
            kl_total += kl_divergence(h2[ch]["hist"], h1[ch]["hist"])
        kl_matrix[(ds1, ds2)] = kl_total / 8  # avg of both directions
        print(f"  KL({ds1:<12}, {ds2:<12}) = {kl_matrix[(ds1, ds2)]:.6f}")

# ── Pairwise Euclidean distance in feature mean space ────────────────────────
print("\n" + "="*70)
print("STEP 9: Feature mean Euclidean distance ...")
print("="*70)
feat_mean_dist = {}
for ds1 in DATASETS:
    for ds2 in DATASETS:
        if ds1 >= ds2:
            continue
        mu1 = np.mean(dataset_features[ds1], axis=0)
        mu2 = np.mean(dataset_features[ds2], axis=0)
        feat_mean_dist[(ds1, ds2)] = float(np.linalg.norm(mu1 - mu2))
        print(f"  ||μ({ds1:<12}) - μ({ds2:<12})|| = {feat_mean_dist[(ds1, ds2)]:.4f}")

# ── Summary: Intra vs Inter ─────────────────────────────────────────────────
seen   = ["recon", "tartan_drive", "sacson", "scand"]
unseen = ["go_stanford"]

intra_fid = [fid_matrix[(a, b)] for a in seen for b in seen if a < b]
inter_fid = [fid_matrix[(a, b)] for a in seen for b in unseen]
cross_fid = [fid_matrix[(a, b)] for a in DATASETS for b in DATASETS if a < b]

print("\n" + "="*70)
print("SUMMARY: Intra-Domain (seen↔seen) vs Inter-Domain (seen↔unseen)")
print("="*70)
print(f"  Intra-Domain FID  mean={np.mean(intra_fid):.2f} ± {np.std(intra_fid):.2f}")
print(f"  Inter-Domain FID mean={np.mean(inter_fid):.2f} ± {np.std(inter_fid):.2f}")
print(f"  Overall FID      mean={np.mean(cross_fid):.2f} ± {np.std(cross_fid):.2f}")

# ── Save raw results ─────────────────────────────────────────────────────────
results = {
    "datasets": DATASETS,
    "n_samples": SAMPLES,
    "fid_matrix": {f"{k[0]}___{k[1]}": v for k, v in fid_matrix.items()},
    "auc_matrix": {f"{k[0]}___{k[1]}": v for k, v in auc_matrix.items()},
    "w1_matrix":  {f"{k[0]}___{k[1]}": v for k, v in w1_matrix.items()},
    "kl_matrix":  {f"{k[0]}___{k[1]}": v for k, v in kl_matrix.items()},
    "feat_mean_dist": {f"{k[0]}___{k[1]}": v for k, v in feat_mean_dist.items()},
    "pixel_stats": dataset_pixel_stats,
    "is_scores": is_scores,
    "summary": {
        "intra_fid_mean": float(np.mean(intra_fid)),
        "intra_fid_std": float(np.std(intra_fid)),
        "inter_fid_mean": float(np.mean(inter_fid)),
        "inter_fid_std": float(np.std(inter_fid)),
    }
}

out_path = Path("/DATA/DATANAS2/xiaoyj25/projects/nwm/distribution_analysis_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("\n✅ All computations done!")
