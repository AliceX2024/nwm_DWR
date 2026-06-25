import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

time_labels = ['1s', '2s', '4s', '8s', '16s']
x = np.arange(len(time_labels))

data = {
    'RECON': {
        'LPIPS': {'pretrained': [3,2,3,1,3], 'ft': [1,3,1,3,1], 'ours': [2,1,2,2,2]},
        'PSNR':  {'pretrained': [2,3,2,1,2], 'ft': [3,2,1,3,1], 'ours': [1,1,3,2,3]},
    },
    'TartanDrive': {
        'LPIPS': {'pretrained': [2,2,3,1,3], 'ft': [1,3,2,2,1], 'ours': [3,1,1,3,2]},
        'PSNR':  {'pretrained': [3,3,3,1,3], 'ft': [1,2,1,3,1], 'ours': [2,1,2,2,2]},
    },
    'SCAND': {
        'LPIPS': {'pretrained': [3,3,3,3,3], 'ft': [1,1,2,2,2], 'ours': [2,2,1,1,1]},
        'PSNR':  {'pretrained': [3,3,1,3,3], 'ft': [1,2,2,1,2], 'ours': [2,1,1,2,1]},
    },
    'HuRoN': {
        'LPIPS': {'pretrained': [3,3,3,3,3], 'ft': [2,1,2,2,1], 'ours': [1,2,1,1,2]},
        'PSNR':  {'pretrained': [3,3,3,3,3], 'ft': [2,2,1,3,1], 'ours': [1,1,2,2,2]},
    },
}

datasets = ['RECON', 'TartanDrive', 'SCAND', 'HuRoN']
metrics  = ['LPIPS', 'PSNR']
GRAY, BLUE, TEAL = '#888780', '#378ADD', '#1D9E75'

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
fig.suptitle('Rank comparison across datasets and time steps\n(rank 1 = best, rank 3 = worst)', fontsize=13)

for row, metric in enumerate(metrics):
    for col, ds in enumerate(datasets):
        ax = axes[row][col]
        d = data[ds][metric]
        ax.axhline(3, color='#E24B4A', linewidth=0.8, linestyle='--', alpha=0.5, label='worst (rank 3)')
        ax.plot(x, d['pretrained'], color=GRAY, marker='o', linewidth=1.8, markersize=5)
        ax.plot(x, d['ft'],         color=BLUE, marker='s', linewidth=1.8, markersize=5)
        ax.plot(x, d['ours'],       color=TEAL, marker='D', linewidth=2.5, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, fontsize=9)
        ax.set_yticks([1,2,3])
        ax.set_ylim(0.5, 3.5)
        ax.invert_yaxis()
        ax.set_title(ds, fontsize=10)
        if col == 0:
            ax.set_ylabel(f'{metric} rank', fontsize=10)
        ax.grid(axis='y', alpha=0.2)

legend_handles = [
    mpatches.Patch(color=GRAY, label='NWM (Pre-trained)'),
    mpatches.Patch(color=BLUE, label='NWM (FT)'),
    mpatches.Patch(color=TEAL, label='Stable-NWM (Ours)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('rank_comparison.pdf', bbox_inches='tight')
plt.show()