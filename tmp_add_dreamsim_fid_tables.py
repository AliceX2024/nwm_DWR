import json
from pathlib import Path

path = Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/significance_analysis.md')
text = path.read_text(encoding='utf-8')

base = json.load(open('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_base_eval_00100000/summary.json', 'r'))
base = next(iter(base.values()))
ft = json.load(open('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_noDWRTrue_eval_0001600/summary.json', 'r'))
ft = next(iter(ft.values()))
ours = json.load(open('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_eval_0001600/summary.json', 'r'))
ours = next(iter(ours.values()))

methods = [
    ('NWM (Pre-trained)', base),
    ('NWM (FT)', ft),
    ('Stable-NWM (Ours)', ours),
]

datasets = ['recon', 'tartan_drive', 'sacson', 'scand']
times = ['1s', '2s', '4s', '8s', '16s']
metrics = ['dreamsim', 'fid']
metric_display = {'dreamsim': 'DreamSim', 'fid': 'FID'}


def fmt(mean, std, rank):
    return '{:.4f}±{:.4f} (rank {})'.format(mean, std, rank)

blocks = []
blocks.append('## 四个数据集的 DreamSim / FID 排名汇总表')
blocks.append('')
blocks.append('- 排名规则：DreamSim 和 FID 都是越低 rank 越高；`rank 1` 表示该时间点该指标下三种方法中的最佳。')
blocks.append('')

for dataset in datasets:
    blocks.append('### {}'.format(dataset))
    blocks.append('')
    blocks.append('| Time | Metric | NWM (Pre-trained) | NWM (FT) | Stable-NWM (Ours) |')
    blocks.append('|---|---|---:|---:|---:|')
    for time in times:
        for metric in metrics:
            key = '{}_time_{}_{}'.format(dataset, metric, time)
            vals = []
            for name, data in methods:
                vals.append((name, data[key]['mean'], data[key]['std']))
            ordered = sorted(vals, key=lambda x: x[1])
            ranks = {name: idx + 1 for idx, (name, _, _) in enumerate(ordered)}
            row = [time, metric_display[metric]]
            for name, mean, std in vals:
                row.append(fmt(mean, std, ranks[name]))
            blocks.append('| {} | {} | {} | {} | {} |'.format(*row))
    blocks.append('')

anchor = '## 总体统计结论\n'
if anchor not in text:
    raise RuntimeError('anchor not found')
new_text = text.replace(anchor, '\n'.join(blocks) + '\n' + anchor, 1)
path.write_text(new_text, encoding='utf-8')
print('updated dreamsim fid tables')
