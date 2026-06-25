import json
import math
from pathlib import Path
from statistics import mean, stdev

OUTPUT = Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/significance_analysis.md')

METHOD_DIRS = {
    'baseline': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_base_eval_00100000'),
    'noDWR': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_noDWRTrue_eval_0001600'),
    'temmeDWR': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_eval_0001600'),
}

HIGHER_BETTER = {'psnr'}
LOWER_BETTER = {'fid', 'lpips', 'dreamsim'}
TIMES = ['1s', '4s', '8s', '16s']
N_EXPECTED = 5


def parse_key(key):
    prefix, metric, time = key.rsplit('_', 2)
    dataset = prefix[:-5] if prefix.endswith('_time') else prefix
    return dataset, metric, time


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_pvalue_from_samples(xs, ys):
    n1 = len(xs)
    n2 = len(ys)
    m1 = mean(xs)
    m2 = mean(ys)
    s1 = stdev(xs) if n1 > 1 else 0.0
    s2 = stdev(ys) if n2 > 1 else 0.0
    se2 = (s1 * s1) / n1 + (s2 * s2) / n2
    if se2 == 0:
        return 1.0 if m1 == m2 else 0.0
    z = abs(m1 - m2) / math.sqrt(se2)
    return 2.0 * (1.0 - normal_cdf(z))


def stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def better(metric, a, b):
    if metric in HIGHER_BETTER:
        return a > b
    if metric in LOWER_BETTER:
        return a < b
    raise ValueError(metric)


def fmt(x):
    return '{:.4f}'.format(x)


def fmt_pm(mean_value, std_value):
    return '{}±{}'.format(fmt(mean_value), fmt(std_value))


def load_run_metrics(method_dir):
    run_dirs = sorted([p for p in method_dir.glob('run_*') if p.is_dir()])
    if len(run_dirs) != N_EXPECTED:
        raise RuntimeError('Expected {} runs in {}, found {}'.format(N_EXPECTED, method_dir, len(run_dirs)))

    collected = {}
    for run_dir in run_dirs:
        inner_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
        if len(inner_dirs) != 1:
            raise RuntimeError('Expected 1 inner dir in {}, found {}'.format(run_dir, len(inner_dirs)))
        inner_dir = inner_dirs[0]
        for json_file in sorted(inner_dir.glob('*_time.json')):
            dataset = json_file.name.replace('_time.json', '')
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, value in data.items():
                parsed_dataset, metric, time = parse_key(key)
                if parsed_dataset != dataset:
                    raise RuntimeError('Dataset mismatch for {} in {}'.format(key, json_file))
                collected.setdefault(dataset, {}).setdefault(time, {}).setdefault(metric, []).append(float(value))

    return collected


raw_runs = {name: load_run_metrics(path) for name, path in METHOD_DIRS.items()}
structured = {}
for method, datasets in raw_runs.items():
    for dataset, times in datasets.items():
        for time, metrics in times.items():
            for metric, values in metrics.items():
                if len(values) != N_EXPECTED:
                    raise RuntimeError('Expected {} values for {} {} {} {}, found {}'.format(N_EXPECTED, method, dataset, time, metric, len(values)))
                structured.setdefault(dataset, {}).setdefault(time, {}).setdefault(metric, {})[method] = {
                    'values': values,
                    'mean': mean(values),
                    'std': stdev(values) if len(values) > 1 else 0.0,
                }

summary_counts = {
    'vs_baseline': {'better': 0, 'sig': 0, 'total': 0},
    'vs_noDWR': {'better': 0, 'sig': 0, 'total': 0},
}

lines = []
lines.append('# 4个数据集 × 4个时间点 × 指标 对比分析')
lines.append('')
lines.append('- p 值说明：基于 5 个不同随机种子的原始 run 结果（每组 n=5），使用 Welch 检验的正态近似计算双侧 p 值。')
lines.append('- 显著性标记：`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` 不显著。')
lines.append('- 指标方向：PSNR 越高越好；FID / LPIPS / DreamSim 越低越好。')
lines.append('- 下表中的 p 值主要比较 `temmeDWR vs baseline` 与 `temmeDWR vs noDWR`。')
lines.append('- 表中均值与标准差由 5 次 run 重新统计得到，并统一四舍五入保留到小数点后四位。')
lines.append('')

for dataset in sorted(structured):
    lines.append('## {}'.format(dataset))
    lines.append('')
    for time in TIMES:
        if time not in structured[dataset]:
            continue
        lines.append('### {}'.format(time))
        lines.append('')
        lines.append('| 指标 | baseline | noDWR | temmeDWR | temme-baseline p | sig | temme-noDWR p | sig | 结论 |')
        lines.append('|---|---:|---:|---:|---:|:---:|---:|:---:|---|')
        for metric in sorted(structured[dataset][time]):
            row = structured[dataset][time][metric]
            b = row['baseline']
            n = row['noDWR']
            t = row['temmeDWR']
            p_tb = welch_pvalue_from_samples(t['values'], b['values'])
            p_tn = welch_pvalue_from_samples(t['values'], n['values'])
            star_tb = stars(p_tb)
            star_tn = stars(p_tn)

            conclusion_parts = []
            if better(metric, t['mean'], b['mean']):
                summary_counts['vs_baseline']['better'] += 1
                conclusion_parts.append('优于baseline')
            else:
                conclusion_parts.append('不优于baseline')
            if p_tb < 0.05:
                summary_counts['vs_baseline']['sig'] += 1
                conclusion_parts.append('vs baseline显著')
            summary_counts['vs_baseline']['total'] += 1

            if better(metric, t['mean'], n['mean']):
                summary_counts['vs_noDWR']['better'] += 1
                conclusion_parts.append('优于noDWR')
            else:
                conclusion_parts.append('不优于noDWR')
            if p_tn < 0.05:
                summary_counts['vs_noDWR']['sig'] += 1
                conclusion_parts.append('vs noDWR显著')
            summary_counts['vs_noDWR']['total'] += 1

            lines.append(
                '| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    metric,
                    fmt_pm(b['mean'], b['std']),
                    fmt_pm(n['mean'], n['std']),
                    fmt_pm(t['mean'], t['std']),
                    fmt(p_tb),
                    star_tb,
                    fmt(p_tn),
                    star_tn,
                    '；'.join(conclusion_parts)
                )
            )
        lines.append('')

lines.append('## 总体统计结论')
lines.append('')
for label, item in summary_counts.items():
    ref = 'baseline' if label == 'vs_baseline' else 'noDWR'
    lines.append(
        '- temmeDWR {}：在 {}/{} 个数据集-时间-指标组合上更优，其中 {}/{} 个达到 p<0.05。'.format(
            label, item['better'], item['total'], item['sig'], item['total']
        )
    )
lines.append('')

records = []
for dataset, times in structured.items():
    for time, metrics in times.items():
        for metric, row in metrics.items():
            t = row['temmeDWR']
            b = row['baseline']
            delta_tb = t['mean'] - b['mean']
            if metric in LOWER_BETTER:
                effect_tb = -delta_tb
            else:
                effect_tb = delta_tb
            p = welch_pvalue_from_samples(t['values'], b['values'])
            records.append((effect_tb, dataset, time, metric, delta_tb, p))

records.sort(reverse=True)
lines.append('## temmeDWR 相对 baseline 的代表性结果')
lines.append('')
lines.append('### 提升最明显的 8 项')
lines.append('')
lines.append('| 数据集 | 时间 | 指标 | temme-baseline 改变量 | p 值 | 显著性 |')
lines.append('|---|---:|---|---:|---:|:---:|')
for effect, dataset, time, metric, delta, p in records[:8]:
    lines.append('| {} | {} | {} | {} | {} | {} |'.format(
        dataset, time, metric, fmt(delta), fmt(p), stars(p)
    ))
lines.append('')
lines.append('### 下降最明显的 8 项')
lines.append('')
lines.append('| 数据集 | 时间 | 指标 | temme-baseline 改变量 | p 值 | 显著性 |')
lines.append('|---|---:|---|---:|---:|:---:|')
for effect, dataset, time, metric, delta, p in records[-8:]:
    lines.append('| {} | {} | {} | {} | {} | {} |'.format(
        dataset, time, metric, fmt(delta), fmt(p), stars(p)
    ))
lines.append('')

OUTPUT.write_text('\n'.join(lines), encoding='utf-8')
print(str(OUTPUT))
