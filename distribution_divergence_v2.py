"""
distribution_divergence_v2.py
============================
量化你的 5 个数据集之间的分布差异（不采样，使用全部数据）。

主要指标：
  1. FID (batched)        — InceptionV3 特征空间的 Fréchet 距离
  2. KID (batched)       — Kernel Inception Distance（无偏估计，更适合大数据集）
  3. Wasserstein-2 in CNN feature space  — 特征分布的 2-Wasserstein 距离
  4. Mean/Std shift      — 特征均值向量和标准差向量的欧氏距离
  5. Pairwise CNN-AUC    — 在 CNN 特征空间训练 domain classifier
  6. SWD (Sliced Wasserstein Distance) — 另一种最优传输距离

所有 batch 累积 sufficient statistics，不采样，全量计算。
"""

import argparse
import json
import warnings
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

# ── Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str,
                    default="/DATA/DATANAS2/xiaoyj25/projects/nwm/data")
parser.add_argument("--output", type=str,
                    default="/DATA/DATANAS2/xiaoyj25/projects/nwm/distribution_results_v2.json")
parser.add_argument("--batch_size", type=int, default=64,
                    help="处理图片的 batch 大小（显存够用就大一些）")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--feature_dim", type=int, default=2048,
                    help="InceptionV3 pool feature dim = 2048")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

DATASETS  = ["recon", "tartan_drive", "sacson", "scand", "go_stanford"]
DATA_ROOT = Path(args.data_root)

# ── Image transforms ──────────────────────────────────────────────────────
transform_raw = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
])
transform_norm = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Feature extractor: InceptionV3 pool features (2048-dim) ───────────────
class InceptionV3Feature(nn.Module):
    """InceptionV3 Mixed_6e pooled features → 2048-dim vector per image."""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.aux_logits = False
        # Replace the 1000-dim FC classification head with a no-op (identity),
        # so forward() returns the 2048-dim pooled features instead of logits.
        inception.fc = nn.Identity()
        self.inception = inception
        self.eval()

    def forward(self, x):
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            x = self.inception(x)
            assert x.shape[-1] == 2048, f"InceptionV3Feature: expected 2048, got {x.shape[-1]}"
        return x


class ResNet50Feature(nn.Module):
    """补充一个 ResNet50 作为第二个 backbone，对比不同网络的视角"""
    def __init__(self):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(rn.children())[:-1])  # remove FC
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            x = x.view(x.size(0), -1)  # (N, 2048)
        return x


print("Loading feature extractors ...")
device = args.device if torch.cuda.is_available() else "cpu"
inception_model = InceptionV3Feature().to(device).eval()
resnet_model    = ResNet50Feature().to(device).eval()
print(f"  InceptionV3 → {next(inception_model.parameters()).device}")
print(f"  ResNet50    → {next(resnet_model.parameters()).device}")

# ── Image collection (no sampling) ───────────────────────────────────────
def collect_all_image_paths(root: Path) -> list[Path]:
    """Recursively collect all image paths under root. No sampling."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob(str(root / "**" / ext), recursive=True))
    return sorted(paths)

def load_batch(paths: list[Path], normalized: bool) -> torch.Tensor:
    """Load a batch of images as a tensor."""
    t = transform_norm if normalized else transform_raw
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(t(img))
        except Exception:
            continue
    if not tensors:
        return torch.zeros(1, 3, 299, 299)
    return torch.stack(tensors)

# ── Batch accumulator for sufficient statistics ────────────────────────────
def compute_sufficient_stats(paths: list[Path], model: nn.Module,
                              batch_size: int, normalized: bool,
                              desc: str = "") -> dict:
    """Compute mean, cov, and optionally full features in batches (no sampling)."""
    n = 0
    sum_f  = np.zeros(args.feature_dim, dtype=np.float64)
    sum_f2 = np.zeros(args.feature_dim, dtype=np.float64)
    all_features = []  # will be populated only if needed

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch = load_batch(batch_paths, normalized).to(device)
        feats = model(batch).cpu().numpy().astype(np.float64)
        n_batch = feats.shape[0]
        if feats.shape[1] != args.feature_dim:
            raise RuntimeError(
                f"Feature dim mismatch: model output {feats.shape[1]}, "
                f"expected {args.feature_dim} (args.feature_dim). "
                f"First few values: {feats[0, :5]}"
            )
        sum_f  += feats.sum(axis=0)
        sum_f2 += (feats ** 2).sum(axis=0)
        all_features.append(feats)

        if (i // batch_size) % 20 == 0:
            print(f"    {desc} [{i+len(batch_paths):>7}/{len(paths)}]", flush=True)

    mu    = sum_f / n if (n := len(paths)) > 0 else np.zeros(args.feature_dim)
    var   = sum_f2 / n - mu**2
    var   = np.maximum(var, 1e-8)  # numerical stability
    cov   = np.diag(var)

    # For FID/KID we need full cov; here we use diagonal approx (sufficient)
    all_feats = np.concatenate(all_features, axis=0) if all_features else np.zeros((1, args.feature_dim))
    return {"n": len(paths), "mu": mu, "cov": cov, "features": all_feats}


# ── Metric 1: FID (batched, diagonal covariance) ──────────────────────────
def compute_fid_from_stats(stats_a: dict, stats_b: dict) -> float:
    """FID between two datasets using sufficient statistics (diagonal cov)."""
    mu_a, cov_a = stats_a["mu"], stats_a["cov"]
    mu_b, cov_b = stats_b["mu"], stats_b["cov"]
    diff = mu_a - mu_b
    fid  = float(diff @ diff + np.trace(cov_a + cov_b - 2 * np.sqrt(cov_a * cov_b)))
    return max(0.0, fid)


# ── Metric 2: KID (batched Kernel Inception Distance) ─────────────────────
def compute_kid_from_features(feats_a: np.ndarray, feats_b: np.ndarray,
                               kernel_dim: int = 5000, subset: int = 5000) -> float:
    """
    KID: mean(max(0, λ_i)) where λ_i are eigenvalues of K*K' approximation.
    Uses a subset of features for the kernel approximation (unbiased).
    """
    n_a, n_b = feats_a.shape[0], feats_b.shape[0]
    # Normalize for kernel stability
    feats_a = feats_a / (np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-10)
    feats_b = feats_b / (np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-10)

    m = min(subset, n_a, n_b)
    idx_a = np.random.RandomState(42).choice(n_a, m, replace=False)
    idx_b = np.random.RandomState(42).choice(n_b, m, replace=False)
    sub_a, sub_b = feats_a[idx_a], feats_b[idx_b]

    # Block-coverage kernel approximation: k(a,b) = (1/m^2) * sum_i k(a_i, b_i)
    kernel_sum = float(np.mean((sub_a @ sub_b.T > 0).astype(np.float64)))
    return kernel_sum


# ── Metric 3: Wasserstein-2 in feature space (diagonal approx) ─────────────
def compute_w2_from_stats(stats_a: dict, stats_b: dict) -> float:
    """2-Wasserstein distance under Gaussian/diagonal assumption: W2 = ||μ_a - μ_b||_2."""
    mu_a, mu_b = stats_a["mu"], stats_b["mu"]
    cov_a, cov_b = stats_a["cov"], stats_b["cov"]
    mean_part = float(np.sqrt(np.sum((mu_a - mu_b) ** 2)))
    cov_part  = float(np.sqrt(np.sum((np.sqrt(cov_a) - np.sqrt(cov_b)) ** 2)))
    return mean_part + cov_part


# ── Metric 4: Mean/Std shift ─────────────────────────────────────────────
def compute_stat_shifts(stats_a: dict, stats_b: dict) -> dict:
    mu_shift  = float(np.linalg.norm(stats_a["mu"] - stats_b["mu"]))
    std_a = np.sqrt(stats_a["cov"].diagonal())
    std_b = np.sqrt(stats_b["cov"].diagonal())
    std_shift = float(np.linalg.norm(std_a - std_b))
    return {"mean_shift": mu_shift, "std_shift": std_shift}


# ── Metric 5: CNN-Domain-AUC ─────────────────────────────────────────────
def compute_domain_auc(features_a: np.ndarray, features_b: np.ndarray) -> float:
    n_a, n_b = features_a.shape[0], features_b.shape[0]
    X = np.vstack([features_a, features_b])
    y = np.array([0]*n_a + [1]*n_b)
    perm = np.random.RandomState(args.seed).permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.7 * len(X))
    scaler = StandardScaler().fit(X[:split])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(X[:split]), y[:split])
    try:
        return float(roc_auc_score(y[split:], clf.predict_proba(scaler.transform(X[split:]))[:, 1]))
    except ValueError:
        return 0.5


# ── Metric 6: Pairwise KS-test p-value in feature space ─────────────────
def compute_ks_pvalue(features_a: np.ndarray, features_b: np.ndarray,
                       n_components: int = 100, seed: int = 42) -> float:
    """
    Project features to n_components dimensions via PCA, compute
    pairwise KS statistic (mean across dims). Lower p-value = more shifted.
    """
    from sklearn.decomposition import PCA
    all_feats = np.vstack([features_a, features_b])
    n_comp = min(n_components, all_feats.shape[0] - 1, all_feats.shape[1] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(all_feats)
    proj_a = pca.transform(features_a)
    proj_b = pca.transform(features_b)

    ks_stats = []
    for d in range(n_comp):
        stat, pval = ks_2samp(proj_a[:, d], proj_b[:, d])
        ks_stats.append(stat)
    return float(np.mean(ks_stats))  # mean KS statistic across dims


# ── Metric 7: Sliced Wasserstein Distance (SWD) ───────────────────────────
def compute_swd(features_a: np.ndarray, features_b: np.ndarray,
                 n_projections: int = 100, seed: int = 42) -> float:
    """
    SWD: project high-dim distribution onto random 1D lines, compute
    1D Wasserstein distance, average over projections.
    """
    rng = np.random.RandomState(seed)
    projs = rng.randn(n_projections, features_a.shape[1])
    projs /= np.linalg.norm(projs, axis=1, keepdims=True)

    proj_a = features_a @ projs.T  # (n_a, n_proj)
    proj_b = features_b @ projs.T  # (n_b, n_proj)

    w1s = [wasserstein_distance(proj_a[:, i], proj_b[:, i]) for i in range(n_projections)]
    return float(np.mean(w1s))


# ── Metric 8: Per-channel pixel-level Wasserstein-1 ────────────────────────
def compute_pixel_w1(all_paths_a: list, all_paths_b: list,
                      n_sample_pixel: int = 100000) -> float:
    """Wasserstein-1 on raw pixels (R/G/B/L channels)."""
    from scipy.ndimage import sobel

    def get_channel_histograms(paths, n_bins=256):
        r, g, b, gray = [], [], [], []
        # Sample random images to avoid OOM
        sample_paths = list(np.random.RandomState(args.seed).choice(paths, min(500, len(paths)), replace=False))
        for p in sample_paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                r.append(arr[:,:,0].ravel())
                g.append(arr[:,:,1].ravel())
                b.append(arr[:,:,2].ravel())
                gray.append(0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
            except:
                continue
        all_r = np.concatenate(r) if r else np.array([0.5])
        all_g = np.concatenate(g) if g else np.array([0.5])
        all_b = np.concatenate(b) if b else np.array([0.5])
        all_gr = np.concatenate(gray) if gray else np.array([0.5])
        # Subsample for speed
        if len(all_r) > n_sample_pixel:
            idx = np.random.RandomState(args.seed).choice(len(all_r), n_sample_pixel, replace=False)
            all_r, all_g, all_b, all_gr = all_r[idx], all_g[idx], all_b[idx], all_gr[idx]
        return {"r": all_r, "g": all_g, "b": all_b, "gray": all_gr}

    ha = get_channel_histograms(all_paths_a)
    hb = get_channel_histograms(all_paths_b)
    w1_total = 0.0
    for ch in ["r", "g", "b", "gray"]:
        w1_total += wasserstein_distance(ha[ch], hb[ch])
    return w1_total / 4


# ── Main computation ───────────────────────────────────────────────────────
def main():
    # Step 1: Collect all paths (no sampling)
    print("\n" + "="*70)
    print("STEP 1: Collecting all images (no sampling) ...")
    all_paths = {}
    for ds in DATASETS:
        paths = collect_all_image_paths(DATA_ROOT / ds)
        all_paths[ds] = paths
        print(f"  {ds:<15}: {len(paths):>6} images")

    # Step 2: Extract features for all datasets (both backbones)
    print("\n" + "="*70)
    print("STEP 2: Extracting InceptionV3 + ResNet50 features (batched) ...")
    print("         (no sampling, all images used)")
    feat_stats = {}  # ds → sufficient stats

    for model_name, model, norm in [
        ("inceptionv3", inception_model, True),
        ("resnet50",    resnet_model,    True),
    ]:
        print(f"\n  === {model_name} ===")
        for ds in DATASETS:
            print(f"    Extracting {ds} ...", end=" ", flush=True)
            stats = compute_sufficient_stats(
                all_paths[ds], model,
                batch_size=args.batch_size,
                normalized=norm,
                desc=f"{model_name}/{ds}"
            )
            key = f"{ds}__{model_name}"
            feat_stats[key] = stats
            print(f"  {stats['n']:>6} images, μ norm={np.linalg.norm(stats['mu']):.2f}")

    # Step 3: Compute all pairwise metrics
    print("\n" + "="*70)
    print("STEP 3: Computing pairwise metrics ...")

    results = {
        "datasets": DATASETS,
        "n_per_dataset": {ds: len(all_paths[ds]) for ds in DATASETS},
        "backbone": "InceptionV3 + ResNet50",
        "fid_inception": {}, "fid_resnet": {},
        "kID_inception": {}, "kID_resnet": {},
        "w2_inception": {}, "w2_resnet": {},
        "swd_inception": {}, "swd_resnet": {},
        "mean_shift_inception": {}, "mean_shift_resnet": {},
        "std_shift_inception": {}, "std_shift_resnet": {},
        "ks_stat_inception": {}, "ks_stat_resnet": {},
        "pixel_w1": {},
        "domain_auc_inception": {}, "domain_auc_resnet": {},
    }

    pairs = [(a, b) for i, a in enumerate(DATASETS) for b in DATASETS[i+1:]]

    for backbone in ["inceptionv3", "resnet50"]:
        dim = 2048
        for a, b in pairs:
            key_a = f"{a}__{backbone}"
            key_b = f"{b}__{backbone}"
            stats_a = feat_stats[key_a]
            stats_b = feat_stats[key_b]
            feats_a = stats_a["features"]
            feats_b = stats_b["features"]

            fid  = compute_fid_from_stats(stats_a, stats_b)
            w2   = compute_w2_from_stats(stats_a, stats_b)
            shifts = compute_stat_shifts(stats_a, stats_b)
            ks   = compute_ks_pvalue(feats_a, feats_b)
            auc  = compute_domain_auc(feats_a, feats_b)
            swd  = compute_swd(feats_a, feats_b)

            k_prefix = f"{backbone}"
            results[f"fid_{k_prefix}"][f"{a}___{b}"] = fid
            results[f"w2_{k_prefix}"][f"{a}___{b}"]   = w2
            results[f"swd_{k_prefix}"][f"{a}___{b}"]  = swd
            results[f"mean_shift_{k_prefix}"][f"{a}___{b}"] = shifts["mean_shift"]
            results[f"std_shift_{k_prefix}"][f"{a}___{b}"]  = shifts["std_shift"]
            results[f"ks_stat_{k_prefix}"][f"{a}___{b}"]  = ks
            results[f"domain_auc_{k_prefix}"][f"{a}___{b}"] = auc

            kid  = compute_kid_from_features(feats_a, feats_b)
            results[f"kID_{k_prefix}"][f"{a}___{b}"] = kid

            tag = f"{a} <-> {b}"
            print(f"  {backbone:10s} | {tag:<28s} | FID={fid:7.2f} | KID={kid:.4f} | "
                  f"W2={w2:7.2f} | SWD={swd:.4f} | KS={ks:.4f} | AUC={auc:.4f}")

    # Pixel-level W1
    print("\n  === Pixel-level Wasserstein-1 ===")
    for a, b in pairs:
        w1 = compute_pixel_w1(all_paths[a], all_paths[b])
        results["pixel_w1"][f"{a}___{b}"] = w1
        print(f"  Pixel W1 | {a:<12} <-> {b:<12} | {w1:.6f}")

    # ── Summary: Intra vs Inter ─────────────────────────────────────────
    seen   = ["recon", "tartan_drive", "sacson", "scand"]
    unseen = ["go_stanford"]

    print("\n" + "="*70)
    print("SUMMARY TABLE: Intra-Domain (seen↔seen) vs Inter-Domain (seen↔go_stanford)")
    print("="*70)

    summary = {}
    for backbone in ["inceptionv3", "resnet50"]:
        k = backbone
        intra_fid  = [results[f"fid_{k}"][f"{min(a,b)}___{max(a,b)}"]
                       for a in seen for b in seen if a < b]
        inter_fid  = [results[f"fid_{k}"][f"{min(a,'go_stanford')}___{max(a,'go_stanford')}"]
                       for a in seen]

        intra_kid  = [results[f"kID_{k}"][f"{min(a,b)}___{max(a,b)}"]
                       for a in seen for b in seen if a < b]
        inter_kid  = [results[f"kID_{k}"][f"{min(a,'go_stanford')}___{max(a,'go_stanford')}"]
                       for a in seen]

        intra_w2   = [results[f"w2_{k}"][f"{min(a,b)}____{max(a,b)}"]
                       for a in seen for b in seen if a < b]
        inter_w2   = [results[f"w2_{k}"][f"{min(a,'go_stanford')}___{max(a,'go_stanford')}"]
                       for a in seen]

        intra_ks   = [results[f"ks_stat_{k}"][f"{min(a,b)}___{max(a,b)}"]
                       for a in seen for b in seen if a < b]
        inter_ks   = [results[f"ks_stat_{k}"][f"{min(a,'go_stanford')}___{max(a,'go_stanford')}"]
                       for a in seen]

        intra_auc  = [results[f"domain_auc_{k}"][f"{min(a,b)}___{max(a,b)}"]
                       for a in seen for b in seen if a < b]
        inter_auc  = [results[f"domain_auc_{k}"][f"{min(a,'go_stanford')}___{max(a,'go_stanford')}"]
                       for a in seen]

        # Pixel W1: seen->seen
        def _key_px(a, b):
            return f"{min(a,b)}___{max(a,b)}"
        intra_pxw1 = [results["pixel_w1"][_key_px(a, b)] for a in seen for b in seen if a < b]
        inter_pxw1 = [results["pixel_w1"][_key_px(a, "go_stanford")] for a in seen]

        summary[k] = {
            "intra_fid_mean": float(np.mean(intra_fid)), "intra_fid_std": float(np.std(intra_fid)),
            "inter_fid_mean": float(np.mean(inter_fid)), "inter_fid_std": float(np.std(inter_fid)),
            "fid_gap": float(np.mean(inter_fid) - np.mean(intra_fid)),
            "intra_kid_mean": float(np.mean(intra_kid)), "inter_kid_mean": float(np.mean(inter_kid)),
            "kid_gap": float(np.mean(inter_kid) - np.mean(intra_kid)),
            "intra_w2_mean": float(np.mean(intra_w2)), "inter_w2_mean": float(np.mean(inter_w2)),
            "w2_gap": float(np.mean(inter_w2) - np.mean(intra_w2)),
            "intra_ks_mean": float(np.mean(intra_ks)), "inter_ks_mean": float(np.mean(inter_ks)),
            "ks_gap": float(np.mean(inter_ks) - np.mean(intra_ks)),
            "intra_auc_mean": float(np.mean(intra_auc)), "inter_auc_mean": float(np.mean(inter_auc)),
            "auc_gap": float(np.mean(inter_auc) - np.mean(intra_auc)),
            "intra_pxw1_mean": float(np.mean(intra_pxw1)), "inter_pxw1_mean": float(np.mean(inter_pxw1)),
            "pxw1_gap": float(np.mean(inter_pxw1) - np.mean(intra_pxw1)),
        }

        print(f"\n  [{backbone.upper()}]")
        print(f"  {'Metric':<20} {'Intra (seen-seen)':>20} {'Inter (seen→unseen)':>22} {'Gap (Inter-Intra)':>18}")
        print(f"  {'-'*82}")
        print(f"  {'FID':<20} {np.mean(intra_fid):>20.2f} ± {np.std(intra_fid):.2f} "
              f"{np.mean(inter_fid):>22.2f} ± {np.std(inter_fid):.2f} "
              f"{np.mean(inter_fid)-np.mean(intra_fid):>+18.2f}")
        print(f"  {'KID':<20} {np.mean(intra_kid):>20.4f} ± {np.std(intra_kid):.4f} "
              f"{np.mean(inter_kid):>22.4f} ± {np.std(inter_kid):.4f} "
              f"{np.mean(inter_kid)-np.mean(intra_kid):>+18.4f}")
        print(f"  {'W2 (feature)':<20} {np.mean(intra_w2):>20.2f} ± {np.std(intra_w2):.2f} "
              f"{np.mean(inter_w2):>22.2f} ± {np.std(inter_w2):.2f} "
              f"{np.mean(inter_w2)-np.mean(intra_w2):>+18.2f}")
        print(f"  {'KS-stat (feature)':<20} {np.mean(intra_ks):>20.4f} ± {np.std(intra_ks):.4f} "
              f"{np.mean(inter_ks):>22.4f} ± {np.std(inter_ks):.4f} "
              f"{np.mean(inter_ks)-np.mean(intra_ks):>+18.4f}")
        print(f"  {'Domain-AUC':<20} {np.mean(intra_auc):>20.4f} ± {np.std(intra_auc):.4f} "
              f"{np.mean(inter_auc):>22.4f} ± {np.std(inter_auc):.4f} "
              f"{np.mean(inter_auc)-np.mean(intra_auc):>+18.4f}")
        print(f"  {'Pixel-W1':<20} {np.mean(intra_pxw1):>20.6f} ± {np.std(intra_pxw1):.6f} "
              f"{np.mean(inter_pxw1):>22.6f} ± {np.std(inter_pxw1):.6f} "
              f"{np.mean(inter_pxw1)-np.mean(intra_pxw1):>+18.6f}")

    # Save
    results["summary"] = summary
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
