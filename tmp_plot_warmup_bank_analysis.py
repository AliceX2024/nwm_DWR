import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"\(step=(\d+)\) Train Loss: ([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training loss and memory-bank occupancy during early training."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to training log.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/warmup_bank_analysis.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=2048,
        help="Memory bank capacity",
    )
    parser.add_argument(
        "--per-step-samples",
        type=int,
        default=128,
        help="How many features are inserted into the bank per training step on one rank",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps before reweighting starts",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=300,
        help="Only plot steps up to this value",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window for loss smoothing; 1 disables smoothing",
    )
    return parser.parse_args()


def moving_average(values, window):
    if window <= 1:
        return values[:]
    smoothed = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        count = min(idx + 1, window)
        smoothed.append(running / count)
    return smoothed


def parse_log(log_path, max_step):
    steps = []
    losses = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            loss = float(match.group(2))
            if step > max_step:
                break
            steps.append(step)
            losses.append(loss)
    if not steps:
        raise RuntimeError(f"No step/loss pairs found in {log_path}")
    return steps, losses


def compute_occupancy(step, bank_size, per_step_samples):
    return min(step * per_step_samples, bank_size)


def main():
    args = parse_args()
    steps, losses = parse_log(args.log, args.max_step)
    losses_smoothed = moving_average(losses, args.smooth_window)

    full_step = math.ceil(args.bank_size / args.per_step_samples)
    occupancies = [compute_occupancy(step, args.bank_size, args.per_step_samples) for step in steps]
    occupancy_ratio = [value / args.bank_size for value in occupancies]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 5.8), dpi=200)
    ax2 = ax1.twinx()

    ax1.plot(steps, losses, color="#9aa0a6", linewidth=1.0, alpha=0.45, label="Train loss (raw)")
    ax1.plot(steps, losses_smoothed, color="#1f77b4", linewidth=2.0, label="Train loss")
    ax2.plot(steps, occupancy_ratio, color="#d62728", linewidth=2.0, linestyle="--", label="Bank occupancy ratio")

    ax1.axvline(full_step, color="#d62728", linestyle=":", linewidth=1.8)
    ax1.text(
        full_step + 1,
        max(losses_smoothed),
        f"bank full @ step {full_step}",
        color="#d62728",
        fontsize=10,
        va="top",
    )

    if args.warmup_steps > 0:
        ax1.axvline(args.warmup_steps, color="#2ca02c", linestyle="-.", linewidth=1.8)
        ax1.text(
            args.warmup_steps + 1,
            min(losses_smoothed),
            f"reweight starts @ step {args.warmup_steps}",
            color="#2ca02c",
            fontsize=10,
            va="bottom",
        )

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Train loss")
    ax2.set_ylabel("Bank occupancy ratio")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_xlim(min(steps), max(steps))
    ax1.set_title("Warm-up / bank-filling analysis")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    summary = (
        f"bank_size={args.bank_size}, per_step_samples={args.per_step_samples}, "
        f"full_step={full_step}, warmup_steps={args.warmup_steps}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.output, bbox_inches="tight")

    print(f"Saved figure to: {args.output}")
    print(f"Parsed {len(steps)} steps from: {args.log}")
    print(f"Bank becomes full at step: {full_step}")
    print(f"Warmup steps: {args.warmup_steps}")


if __name__ == "__main__":
    main()
