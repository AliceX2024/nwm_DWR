"""
distribution_divergence.py
==========================
量化你的 5 个数据集之间的分布差异：
  - recon, tartan_drive, sacson, scand  (训练时见过 → Intra-Domain)
  - go_stanford                            (训练时没见过 → Zero-Shot / Inter-Domain)

输出：
  1. Pairwise FID 矩阵
  2. Pairwise Domain-Classifier AUC
  3. Per-dataset IS (Inception Score)
  4. Summary: Intra vs Inter domain 对比

依赖（均已在你的 conda 环境中）：
  torch, torcheval, numpy, scipy, scikit-learn
  PIL (Pillow), torchvision

运行示例：
  conda activate <your_env>
  python distribution_divergence.py --samples 500 --batch_size 32
"""

import argparse
import json
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import torch
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str,
                    default="/DATA/DATANAS2/xiaoyj25/projects/nwm/data")
parser.add_argument("--output", type=str,
                    default="/DATA/DATANAS2/xiaoyj25/projects/nwm/distribution_results.json")
parser.add_argument("--samples", type=int, default=500,
                    help="每数据集最多采样多少张图")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

DATASETS = ["recon", "tartan_drive", "sacson", "scand", "go_stanford"]
DATA_ROOT = Path(args.data_root)

# ── Two transforms:
#   1. raw [0,1]  → for torcheval FID (expects [0, 1])
#   2. normalized → for torchvision InceptionV3 feature extraction (domain-auc, IS)
transform_raw = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),          # [0, 1] float32
])

transform_norm = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images(root: Path, max_n: int, normalized: bool = False) -> torch.Tensor:
    """Load up to max_n images from a dataset root directory.

    Args:
        root: dataset root directory.
        max_n: max number of images to load.
        normalized: if True, apply ImageNet normalization (for Inception feature extraction).
                    if False, keep raw [0,1] (for torcheval FID).
    """
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
        paths.extend(glob(str(root / "**" / ext), recursive=True))
    if len(paths) > max_n:
        paths = list(np.random.choice(paths, max_n, replace=False))

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

# ── FID: use torcheval's built-in InceptionV3 ────────────────────────────
def compute_pairwise_fid(data_root: Path, datasets: list, max_n: int,
                         batch_size: int, device: str) -> dict:
    print("=== Computing Pairwise FID ===")
    fid_matrix = {}

    # Pre-load all datasets (raw [0,1] → for torcheval FID)
    all_tensors = {}
    for ds in datasets:
        print(f"  Loading {ds} ...", end=" ", flush=True)
        tensors = load_images(data_root / ds, max_n, normalized=False).to(device)
        all_tensors[ds] = tensors
        print(f"{tensors.shape[0]} images")

    # Pairwise FID
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            if j <= i:
                continue
            fid = FrechetInceptionDistance(feature_dim=2048).to(device)
            real_batch = all_tensors[ds1]
            fake_batch = all_tensors[ds2]

            for k in range(0, len(real_batch), batch_size):
                fid.update(real_batch[k:k+batch_size], is_real=True)
            for k in range(0, len(fake_batch), batch_size):
                fid.update(fake_batch[k:k+batch_size], is_real=False)

            score = fid.compute().item()
            fid_matrix[f"{ds1}___{ds2}"] = score
            print(f"  FID({ds1:<14}, {ds2:<14}) = {score:.4f}")
            fid.reset()

    return fid_matrix

# ── Domain-Classifier AUC ─────────────────────────────────────────────────
def compute_domain_auc(data_root: Path, datasets: list, max_n: int,
                        device: str) -> dict:
    """Extract Inception features, then train LogisticRegression to distinguish pairs."""
    print("\n=== Computing Domain-Classifier AUC ===")
    # We'll use torcheval's FID metric but grab the features ourselves via a manual pass
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    # Extract features
    features = {}
    for ds in datasets:
        print(f"  Extracting features for {ds} ...", end=" ", flush=True)
        tensors = load_images(data_root / ds, max_n, normalized=True)  # normalized=True for InceptionV3
        feats = []
        with torch.no_grad():
            for i in range(0, len(tensors), 32):
                x = tensors[i:i+32].to(device)
                f = model(x).cpu().numpy()
                feats.append(f)
        features[ds] = np.concatenate(feats, axis=0)
        print(f"  shape={features[ds].shape}")

    auc_matrix = {}
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            if j <= i:
                continue
            f1, f2 = features[ds1], features[ds2]
            X = np.vstack([f1, f2])
            y = np.array([0]*len(f1) + [1]*len(f2))
            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]
            split = int(0.7 * len(X))
            X_tr, y_tr = X[:split], y[:split]
            X_te, y_te = X[split:], y[split:]
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, proba)
            except ValueError:
                auc = 0.5
            auc_matrix[f"{ds1}___{ds2}"] = float(auc)
            print(f"  AUC({ds1:<14}, {ds2:<14}) = {auc:.4f}")

    return auc_matrix

# ── Inception Score (approximation: mean pairwise cosine distance) ────────
def compute_is_diversity(data_root: Path, datasets: list, max_n: int,
                          device: str) -> dict:
    """Higher = more diverse feature distribution within dataset."""
    print("\n=== Computing Inception Score (diversity proxy) ===")
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    is_scores = {}
    for ds in datasets:
        tensors = load_images(data_root / ds, max_n, normalized=True)  # InceptionV3 expects normalized input
        feats = []
        with torch.no_grad():
            for i in range(0, len(tensors), 32):
                f = model(tensors[i:i+32].to(device)).cpu().numpy()
                feats.append(f)
        feat = np.concatenate(feats, axis=0)
        # Mean pairwise cosine distance as diversity proxy
        norms = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-10)
        cos_sim = norms @ norms.T
        triu_idx = np.triu_indices(cos_sim.shape[0], k=1)
        score = float(np.mean(1.0 - cos_sim[triu_idx]))
        is_scores[ds] = score
        print(f"  IS({ds:<15}) = {score:.4f}")

    return is_scores

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    fid_matrix  = compute_pairwise_fid(DATA_ROOT, DATASETS, args.samples, args.batch_size, device)
    auc_matrix   = compute_domain_auc(DATA_ROOT, DATASETS, args.samples, device)
    is_scores    = compute_is_diversity(DATA_ROOT, DATASETS, args.samples, device)

    # ── Summary: Intra vs Inter ───────────────────────────────────────────
    seen   = ["recon", "tartan_drive", "sacson", "scand"]
    unseen = ["go_stanford"]

    def _fid_key(a, b):
        """Match the lexicographic ordering used during storage."""
        return f"{min(a, b)}___{max(a, b)}"

    def _auc_key(a, b):
        return f"{min(a, b)}___{max(a, b)}"

    intra_fid = [fid_matrix[_fid_key(a, b)]
                 for a in seen for b in seen if a < b]
    inter_fid = [fid_matrix[_fid_key(a, b)]
                 for a in seen for b in unseen]

    intra_auc = [auc_matrix[_auc_key(a, b)]
                 for a in seen for b in seen if a < b]
    inter_auc = [auc_matrix[_auc_key(a, "go_stanford")]
                 for a in seen]

    print("\n" + "="*60)
    print("SUMMARY: Intra-Domain (seen↔seen) vs Inter-Domain (seen↔go_stanford)")
    print("="*60)
    print(f"  Intra-Domain FID  mean={np.mean(intra_fid):.2f} ± {np.std(intra_fid):.2f}")
    print(f"  Inter-Domain FID mean={np.mean(inter_fid):.2f} ± {np.std(inter_fid):.2f}")
    print(f"  FID Gap (Inter - Intra) = {np.mean(inter_fid) - np.mean(intra_fid):.2f}")
    print()
    print(f"  Intra-Domain AUC  mean={np.mean(intra_auc):.4f} ± {np.std(intra_auc):.4f}")
    print(f"  Inter-Domain AUC mean={np.mean(inter_auc):.4f} ± {np.std(inter_auc):.4f}")
    print(f"  AUC Gap (Inter - Intra) = {np.mean(inter_auc) - np.mean(intra_auc):.4f}")
    print()

    results = {
        "datasets": DATASETS,
        "n_samples_per_dataset": args.samples,
        "fid_matrix": fid_matrix,
        "auc_matrix": auc_matrix,
        "is_scores": is_scores,
        "summary": {
            "intra_fid_mean": float(np.mean(intra_fid)),
            "intra_fid_std": float(np.std(intra_fid)),
            "inter_fid_mean": float(np.mean(inter_fid)),
            "inter_fid_std": float(np.std(inter_fid)),
            "fid_gap": float(np.mean(inter_fid) - np.mean(intra_fid)),
            "intra_auc_mean": float(np.mean(intra_auc)),
            "inter_auc_mean": float(np.mean(inter_auc)),
            "auc_gap": float(np.mean(inter_auc) - np.mean(intra_auc)),
        }
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()