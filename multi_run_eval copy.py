#!/usr/bin/env python3
"""
multi_run_eval.py
=================
外层 wrapper，对同一实验配置多次调用 isolated_nwm_infer.py + isolated_nwm_eval.py，
最后汇总所有 run 的指标，输出均值 ± 标准差。

用法示例：
    python multi_run_eval.py \
        --exp config/your_experiment.yaml \
        --datasets go_stanford \
        --eval_type time \
        --gt_dir eval_results/gt \
        --n_runs 5 \
        --base_output_dir eval_results/multi_run

    # 或同时跑 rollout：
    python multi_run_eval.py \
        --exp config/your_experiment.yaml \
        --datasets go_stanford \
        --eval_type time,rollout \
        --gt_dir eval_results/gt \
        --n_runs 5 \
        --base_output_dir eval_results/multi_run

推理过程中涉及随机性的地方只有两个，都在 isolated_nwm_infer.py 的 model_forward_wrapper 里：
# ❶ 扩散模型初始噪声（DDPM/DDIM 反向过程的起点）
z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
# ❷ VAE 编码时的随机采样（latent_dist.sample()）
x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
--seed 设置后，两个随机源都被固定——相同的 seed 产生完全相同的推理结果，不同的 seed 产生不同的图像。这就是 5 次独立实验的差异来源。

python multi_run_eval.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval.yaml \
    --ckp 0001600 \
    --datasets go_stanford \
    --eval_type time \
    --gt_dir eval_results/gt \
    --base_output_dir eval_results/multi_run_0001600 \
    --n_runs 5 \
    --base_seed 0 \
    --ngpus 1 \
    --batch_size 64 \
    --num_workers 12

python multi_run_eval.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval.yaml \
    --ckp 0001600 \
    --datasets go_stanford \
    --eval_type rollout \
    --gt_dir eval_results/gt \
    --base_output_dir eval_results/multi_run_0001600 \
    --n_runs 5 \
    --base_seed 0 \
    --ngpus 1 \
    --batch_size 64 \
    --num_workers 12
"""

import argparse
import os
import json
import subprocess
import shutil
import numpy as np
from pathlib import Path
from glob import glob


def run_cmd(cmd, env=None, check=True):
    """Run a command, print it, and raise on failure."""
    print(f"\n{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def run_infer(run_dir, args, run_seed):
    """Run isolated_nwm_infer.py once for a given seed and output dir."""
    import random
    master_port = random.randint(20000, 60000)

    cmd = [
        "torchrun",
        "--nproc_per_node", str(args.ngpus),
        "--master_port", str(master_port),
        "isolated_nwm_infer.py",
        "--output_dir", run_dir,
        "--exp", args.exp,
        "--ckp", args.ckp,
        "--datasets", args.datasets,
        "--eval_type", args.eval_type,
        "--num_sec_eval", str(args.num_sec_eval),
        "--input_fps", str(args.input_fps),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--rollout_fps_values", args.rollout_fps_values,
        "--gt", str(args.gt),
        # 关键：每次推理用不同 seed，驱动 torch.randn 的随机性
        "--seed", str(run_seed),
    ]

    env = os.environ.copy()
    run_cmd(cmd, env=env)


def run_eval(args, run_dir):
    """
    Run isolated_nwm_eval.py once on a given inference output dir.

    isolated_nwm_infer.py always writes under output_dir, creating a subdirectory:
        output_dir / {run_name} / {ckp} / {dataset} / eval_type / id_*/
    isolated_nwm_eval.py expects:
        exp_dir / {dataset} / eval_type / id_*/
    Therefore eval_exp_dir = run_dir / {run_name} / {ckp} (one level above dataset).
    This is identical for both --gt 0 and --gt 1 since infer uses --exp in both cases.
    """
    import yaml
    with open(args.exp) as f:
        cfg = yaml.safe_load(f)
    run_name = cfg.get("run_name", Path(args.exp).stem)
    eval_exp_dir = str(Path(run_dir) / run_name / args.ckp)

    eval_types_str = args.eval_type.replace(",", ",")  # keep as-is
    cmd = [
        "python", "isolated_nwm_eval.py",
        "--gt_dir", args.gt_dir,
        "--exp_dir", eval_exp_dir,
        "--datasets", args.datasets,
        "--eval_types", eval_types_str,
        "--num_sec_eval", str(args.num_sec_eval),
        "--input_fps", str(args.input_fps),
        "--batch_size", str(args.batch_size),
        "--rollout_fps_values", args.rollout_fps_values,
    ]

    env = os.environ.copy()
    run_cmd(cmd, env=env)


def aggregate_results(args):
    """
    Collect all per-run JSON files under base_output_dir,
    compute mean ± std for every metric, write to summary JSON.
    """
    base = Path(args.base_output_dir)

    # Discover all run subdirs (format: run_000, run_001, ...)
    run_dirs = sorted(base.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* subdirs found under {base}")

    # ── Discover all dataset/eval-type groups ────────────────────────
    # Walk once to build {group_key: {metric: [values across runs]}}
    agg = {}          # {group_key: {metric_key: [float, ...]}}

    for run_dir in run_dirs:
        run_jsons = list(run_dir.rglob("*.json"))
        if not run_jsons:
            print(f"[WARN] No JSON found in {run_dir}, skipping")
            continue

        for json_path in run_jsons:
            rel = json_path.relative_to(run_dir)           # e.g. "go_stanford/time/go_stanford_time.json"
            parent_parts = rel.parent.parts                 # e.g. ("go_stanford", "time")
            group_key = "/".join(parent_parts) if parent_parts else "__root__"

            with open(json_path) as f:
                scores = json.load(f)

            for k, v in scores.items():
                agg.setdefault(group_key, {}).setdefault(k, []).append(v)

    # Compute mean and std
    summary = {}
    for group_key, metrics in sorted(agg.items()):
        print(f"\n{'='*60}")
        print(f"[{group_key}]")
        summary[group_key] = {}
        for metric, values in sorted(metrics.items()):
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            summary[group_key][metric] = {"mean": mean, "std": std}
            n = len(arr)
            print(f"  {metric}: {mean:.6f} ± {std:.6f}  (n={n}, raw={values})")

    # Write summary
    summary_path = base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {summary_path}")

    # Also print a markdown-friendly table
    print("\n" + "="*60)
    print("Markdown table (mean ± std)")
    print("="*60)
    for group_key, metrics in sorted(summary.items()):
        print(f"\n**{group_key}**")
        print(f"| Metric | Mean | Std |")
        print(f"|--------|------|-----|")
        for metric, stat in sorted(metrics.items()):
            print(f"| {metric} | {stat['mean']:.6f} | {stat['std']:.6f} |")

    return summary


def main(args):
    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # GT does not need repeated runs; check whether the eval JSON already exists
    if args.gt:
        # eval writes JSONs to gt_dir/{dataset}/  e.g. eval_results/gt/go_stanford/go_stanford_time.json
        first_dataset = args.datasets.split(",")[0]
        gt_json = Path(args.gt_dir) / first_dataset / f"{first_dataset}_{args.eval_type}.json"
        if not gt_json.exists():
            print(f"[INFO] GT baseline not found at {gt_json}, generating once...")
            gt_run_dir = args.gt_dir
            run_infer(gt_run_dir, args, run_seed=0)
            run_eval(args, gt_run_dir)
        else:
            print(f"[INFO] GT baseline found at {gt_json}, skipping generation.")

    # Multiple inference + eval runs
    for i in range(args.n_runs):
        run_idx = f"run_{i:03d}"
        run_seed = args.base_seed + i
        run_dir = str(base_output_dir / run_idx)

        print(f"\n{'#'*60}")
        print(f"# Run {i+1}/{args.n_runs}  |  seed={run_seed}  |  output={run_dir}")
        print(f"{'#'*60}")

        # Clean output dir if exists (otherwise eval overwrites old results)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

        # 1. Inference
        run_infer(run_dir, args, run_seed)

        # 2. Evaluation
        run_eval(args, run_dir)

    # 3. Aggregate
    print(f"\n\n{'#'*60}")
    print("# Aggregating results across {args.n_runs} runs")
    print(f"{'#'*60}")
    aggregate_results(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run isolated_nwm_infer + isolated_nwm_eval multiple times "
                    "and report mean ± std."
    )

    # ── Experiment config ──────────────────────────────────────────
    parser.add_argument("--exp", type=str, required=True,
                        help="Path to experiment YAML config (passed to isolated_nwm_infer)")
    parser.add_argument("--ckp", type=str, default="0100000",
                        help="Checkpoint name suffix (default: 0100000)")
    parser.add_argument("--gt", type=int, default=0,
                        help="Set to 1 to generate GT baseline")

    # ── Data ──────────────────────────────────────────────────────
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated dataset names, e.g. 'go_stanford' or 'go_stanford,reconbench'")
    parser.add_argument("--gt_dir", type=str, default="eval_results/gt",
                        help="Output dir for GT baseline (only used if --gt 1)")

    # ── Evaluation type ────────────────────────────────────────────
    parser.add_argument("--eval_type", type=str, default="time",
                        help="'time', 'rollout', or comma-separated 'time,rollout'")
    parser.add_argument("--num_sec_eval", type=int, default=5,
                        help="Number of time-of-day evaluation points (2^i, i=0..N-1)")
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--rollout_fps_values", type=str, default="1,4")

    # ── Multi-run settings ────────────────────────────────────────
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of independent inference+eval runs (default: 5, matching baseline)")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Starting seed; runs use base_seed, base_seed+1, ... (default: 0)")
    parser.add_argument("--base_output_dir", type=str, default="eval_results/multi_run",
                        help="Parent directory that will contain run_000/, run_001/, ...")

    # ── Hardware ──────────────────────────────────────────────────
    parser.add_argument("--ngpus", type=int, default=1,
                        help="Number of GPUs for torchrun (default: 1, matching EVAL_GUIDE.md)")

    # ── Dataloader ────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parsed = parser.parse_args()

    # Normalise eval_type (it is passed to isolated_nwm_eval.py --eval_types)
    # and also used to set --eval_type for isolated_nwm_infer.py
    # The infer script only accepts one eval_type, so pick the first
    parsed.eval_type = parsed.eval_type.split(",")[0]

    main(parsed)
