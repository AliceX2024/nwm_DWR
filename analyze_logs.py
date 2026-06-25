#!/usr/bin/env python3
"""
分析并可视化训练日志中的 Loss 和 Reweight 权重统计信息
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
from pathlib import Path

def parse_log_file(log_path):
    """解析日志文件，提取训练损失和权重统计信息"""
    
    train_loss_pattern = r'\(step=(\d+)\)\s+Train Loss:\s+([\d.]+)'
    
    train_losses = []
    reweight_stats = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 解析 Train Loss
        for match in re.finditer(train_loss_pattern, content):
            step = int(match.group(1))
            loss = float(match.group(2))
            train_losses.append((step, loss))
        
        # 解析 Reweight Debug - 匹配整个块
        # 格式: [Reweight Debug Step X]
        #       Stats: Mean=..., Std=...
        #       Relation: Intra-Traj Std=... (内), Inter-Traj Std=... (间)
        reweight_block_pattern = r'\[Reweight Debug Step (\d+)\]\s*\n.*?Stats: Mean=([\d.]+), Std=([\d.]+)\s*\n.*?Intra-Traj Std=([\d.]+).*?Inter-Traj Std=([\d.]+)'
        
        for match in re.finditer(reweight_block_pattern, content, re.DOTALL):
            step = int(match.group(1))
            mean = float(match.group(2))
            std = float(match.group(3))
            intra_std = float(match.group(4))
            inter_std = float(match.group(5))
            reweight_stats.append({
                'step': step,
                'mean': mean,
                'std': std,
                'intra_std': intra_std,
                'inter_std': inter_std
            })
    
    print(f"  [DEBUG] Found {len(train_losses)} train losses, {len(reweight_stats)} reweight stats")
    
    return train_losses, reweight_stats


def smooth_data(data, window=5):
    """对数据进行滑动平均平滑"""
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


def plot_comparison(results_dict, output_dir):
    """绘制对比图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    # 1. 训练 Loss 对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 Loss 曲线对比
    ax = axes[0, 0]
    for i, (name, data) in enumerate(results_dict.items()):
        if data['train_losses']:
            steps = [x[0] for x in data['train_losses']]
            losses = [x[1] for x in data['train_losses']]
            ax.plot(steps, losses, label=name, alpha=0.7, linewidth=1.5, color=colors[i % len(colors)])
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 权重 Std 对比
    ax = axes[0, 1]
    for i, (name, data) in enumerate(results_dict.items()):
        if data['reweight_stats']:
            steps = [x['step'] for x in data['reweight_stats']]
            stds = [x['std'] for x in data['reweight_stats']]
            ax.plot(steps, stds, label=name, alpha=0.7, linewidth=1.5, color=colors[i % len(colors)])
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Weight Std')
    ax.set_title('Reweight Std over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 轨迹内 vs 轨迹间方差对比
    ax = axes[1, 0]
    for i, (name, data) in enumerate(results_dict.items()):
        if data['reweight_stats']:
            steps = [x['step'] for x in data['reweight_stats']]
            intra_stds = [x['intra_std'] for x in data['reweight_stats']]
            inter_stds = [x['inter_std'] for x in data['reweight_stats']]
            ax.plot(steps, intra_stds, label=f'{name} (Intra)', alpha=0.7, linewidth=1.5, color=colors[i % len(colors)])
            ax.plot(steps, inter_stds, label=f'{name} (Inter)', alpha=0.7, linewidth=1.5, linestyle='--', color=colors[i % len(colors)])
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Std')
    ax.set_title('Intra-Traj vs Inter-Traj Std')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.4 权重分布箱线图
    ax = axes[1, 1]
    all_stds = []
    labels = []
    for name, data in results_dict.items():
        if data['reweight_stats']:
            stds = [x['std'] for x in data['reweight_stats']]
            all_stds.append(stds)
            labels.append(name)
    
    if all_stds:
        bp = ax.boxplot(all_stds, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(all_stds)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax.set_ylabel('Weight Std')
    ax.set_title('Weight Std Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_overview.png")
    
    # 2. 单独绘制每个实验的详细分析
    for name, data in results_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Analysis: {name}', fontsize=14, fontweight='bold')
        
        # 2.1 Loss 曲线
        ax = axes[0, 0]
        if data['train_losses']:
            steps = [x[0] for x in data['train_losses']]
            losses = [x[1] for x in data['train_losses']]
            # 平滑
            if len(losses) > 10:
                losses_smooth = smooth_data(losses, window=10)
                ax.plot(steps[:len(losses_smooth)], losses_smooth, linewidth=2, color=colors[0])
                ax.scatter(steps, losses, alpha=0.3, s=10, color=colors[0])
            else:
                ax.plot(steps, losses, linewidth=2, color=colors[0])
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Train Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # 2.2 权重统计
        ax = axes[0, 1]
        if data['reweight_stats']:
            steps = [x['step'] for x in data['reweight_stats']]
            means = [x['mean'] for x in data['reweight_stats']]
            stds = [x['std'] for x in data['reweight_stats']]
            
            ax.plot(steps, means, label='Mean', linewidth=2, color='blue')
            ax.fill_between(steps, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.3, color='blue', label='Mean ± Std')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weight')
        ax.set_title('Weight Mean & Std')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2.3 轨迹内/间方差
        ax = axes[1, 0]
        if data['reweight_stats']:
            steps = [x['step'] for x in data['reweight_stats']]
            intra_stds = [x['intra_std'] for x in data['reweight_stats']]
            inter_stds = [x['inter_std'] for x in data['reweight_stats']]
            
            ax.plot(steps, intra_stds, label='Intra-Traj Std', linewidth=2, color='green')
            ax.plot(steps, inter_stds, label='Inter-Traj Std', linewidth=2, color='orange')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Std')
        ax.set_title('Intra vs Inter Trajectory Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2.4 统计摘要
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "=== Training Summary ===\n\n"
        if data['train_losses']:
            losses = [x[1] for x in data['train_losses']]
            summary_text += f"Loss:\n"
            summary_text += f"  - Initial: {losses[0]:.4f}\n"
            summary_text += f"  - Final: {losses[-1]:.4f}\n"
            summary_text += f"  - Min: {min(losses):.4f}\n"
            summary_text += f"  - Max: {max(losses):.4f}\n"
            summary_text += f"  - Mean: {np.mean(losses):.4f}\n"
        
        if data['reweight_stats']:
            stds = [x['std'] for x in data['reweight_stats']]
            intra_stds = [x['intra_std'] for x in data['reweight_stats']]
            inter_stds = [x['inter_std'] for x in data['reweight_stats']]
            
            summary_text += f"\nWeight Statistics:\n"
            summary_text += f"  - Std Mean: {np.mean(stds):.4f}\n"
            summary_text += f"  - Std Std: {np.std(stds):.4f}\n"
            summary_text += f"  - Intra-Traj Mean: {np.mean(intra_stds):.4f}\n"
            summary_text += f"  - Inter-Traj Mean: {np.mean(inter_stds):.4f}\n"
            summary_text += f"  - Intra/Inter Ratio: {np.mean(intra_stds)/max(np.mean(inter_stds), 1e-6):.3f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        safe_name = name.replace('/', '_').replace(' ', '_')
        plt.savefig(f'{output_dir}/analysis_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/analysis_{safe_name}.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('--logs', nargs='+', required=True, help='Log file paths')
    parser.add_argument('--output', type=str, default='./log_analysis', help='Output directory')
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 解析所有日志文件
    results = {}
    for log_path in args.logs:
        name = Path(log_path).stem  # 使用文件名作为实验名
        # 如果有父目录，也加入命名
        parent_name = Path(log_path).parent.name
        if parent_name and parent_name != 'logs':
            name = f"{parent_name}"
        
        print(f"Parsing: {log_path}")
        train_losses, reweight_stats = parse_log_file(log_path)
        
        results[name] = {
            'train_losses': train_losses,
            'reweight_stats': reweight_stats
        }
        
        print(f"  - Train losses: {len(train_losses)} entries")
        print(f"  - Reweight stats: {len(reweight_stats)} entries")
    
    # 绘制对比图
    print("\nGenerating comparison plots...")
    plot_comparison(results, args.output)
    
    # 打印对比摘要
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for name, data in results.items():
        print(f"\n### {name} ###")
        
        if data['train_losses']:
            losses = [x[1] for x in data['train_losses']]
            print(f"Loss: Initial={losses[0]:.4f}, Final={losses[-1]:.4f}, Min={min(losses):.4f}")
        
        if data['reweight_stats']:
            stds = [x['std'] for x in data['reweight_stats']]
            intra_stds = [x['intra_std'] for x in data['reweight_stats']]
            inter_stds = [x['inter_std'] for x in data['reweight_stats']]
            print(f"Weight Std: Mean={np.mean(stds):.4f}, Std={np.std(stds):.4f}")
            print(f"Intra-Traj: Mean={np.mean(intra_stds):.4f}")
            print(f"Inter-Traj: Mean={np.mean(inter_stds):.4f}")
            print(f"Intra/Inter Ratio: {np.mean(intra_stds)/max(np.mean(inter_stds), 1e-6):.3f}")
    
    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
