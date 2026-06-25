import json
import math
from pathlib import Path
from statistics import mean, stdev

OUTPUT = Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/ablation_bank_strategy_significance_analysis.md')

METHOD_DIRS = {
    'baseline': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_base_eval_00100000'),
    'noDWR': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_noDWRTrue_eval_0001600'),
    'temmeDWR_bank2048': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_eval_0001600'),
    'temmeDWR_fifo': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_fifo_eval_0001600'),
    'temmeDWR_random': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_random_eval_0001600'),
    'temmeDWR_random_reweight': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_random_reweight_eval_0001600'),
    'latent_only_bank0': Path('/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_bank0_nopool_notime_eval_0001600'),
}

DISPLAY_NAMES = {
    'baseline': 'baseline',
    'noDWR': 'noDWR',
    'temmeDWR_bank2048': 'temmeDWR(bank2048)',
    'temmeDWR_fifo': 'temmeDWR(fifo)',
    'temmeDWR_random': 'temmeDWR(random)',
    'temmeDWR_random_reweight': 'temmeDWR(random_reweight)',
    'latent_only_bank0': 'latent_only(bank0)',
}

ABLATIONS = [
    'temmeDWR_bank2048',
    'temmeDWR_fifo',
    'temmeDWR_random',
    'temmeDWR_random_reweight',
    'latent_only_bank0',
]
COMPARE_TO = ['baseline', 'noDWR', 'temmeDWR_bank2048']
ALL_METHODS = [
    'baseline',
    'noDWR',
    'temmeDWR_bank2048',
    'temmeDWR_fifo',
    'temmeDWR_random',
    'temmeDWR_random_reweight',
    'latent_only_bank0',
]
HIGHER_BETTER = {'psnr'}
LOWER_BETTER = {'fid', 'lpips', 'dreamsim'}
TIMES = ['1s', '2s', '4s', '8s', '16s']
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

summary_counts = {}
for method in ABLATIONS:
    for ref in COMPARE_TO:
        if method == ref:
            continue
        summary_counts['{}_vs_{}'.format(method, ref)] = {'better': 0, 'sig': 0, 'total': 0}

lines = []
lines.append('# 消融实验：4个数据集 × 5个时间点 × 指标 对比分析（bank 替换策略）')
lines.append('')
lines.append('- p 值说明：基于 5 个不同随机种子的原始 run 结果（每组 n=5），使用 Welch 检验的正态近似计算双侧 p 值。')
lines.append('- 显著性标记：`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` 不显著。')
lines.append('- 指标方向：PSNR 越高越好；FID / LPIPS / DreamSim 越低越好。')
lines.append('- 比较对象包含 baseline、noDWR、temmeDWR(bank2048) 以及不同 bank 替换/重加权策略与 latent-only 变体。')
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
        lines.append('| 指标 | baseline | noDWR | bank2048 | fifo | random | random_reweight | latent_only(bank0) | 结论 |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        for metric in sorted(structured[dataset][time]):
            row = structured[dataset][time][metric]
            ranked = sorted(ALL_METHODS, key=lambda m: row[m]['mean'], reverse=(metric in HIGHER_BETTER))
            best_method = ranked[0]
            conclusion = '最佳方法：{}'.format(DISPLAY_NAMES[best_method])
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                metric,
                fmt_pm(row['baseline']['mean'], row['baseline']['std']),
                fmt_pm(row['noDWR']['mean'], row['noDWR']['std']),
                fmt_pm(row['temmeDWR_bank2048']['mean'], row['temmeDWR_bank2048']['std']),
                fmt_pm(row['temmeDWR_fifo']['mean'], row['temmeDWR_fifo']['std']),
                fmt_pm(row['temmeDWR_random']['mean'], row['temmeDWR_random']['std']),
                fmt_pm(row['temmeDWR_random_reweight']['mean'], row['temmeDWR_random_reweight']['std']),
                fmt_pm(row['latent_only_bank0']['mean'], row['latent_only_bank0']['std']),
                conclusion
            ))
        lines.append('')

        for metric in sorted(structured[dataset][time]):
            row = structured[dataset][time][metric]
            lines.append('#### {} / {} / 显著性比较'.format(time, metric))
            lines.append('')
            lines.append('| 方法 | 对比对象 | 方法均值±标准差 | 对比均值±标准差 | p 值 | 显著性 | 是否更优 |')
            lines.append('|---|---|---:|---:|---:|:---:|:---:|')
            for method in ABLATIONS:
                for ref in COMPARE_TO:
                    if method == ref:
                        continue
                    m = row[method]
                    r = row[ref]
                    p = welch_pvalue_from_samples(m['values'], r['values'])
                    is_better = better(metric, m['mean'], r['mean'])
                    key = '{}_vs_{}'.format(method, ref)
                    if is_better:
                        summary_counts[key]['better'] += 1
                    if p < 0.05:
                        summary_counts[key]['sig'] += 1
                    summary_counts[key]['total'] += 1
                    lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                        DISPLAY_NAMES[method],
                        DISPLAY_NAMES[ref],
                        fmt_pm(m['mean'], m['std']),
                        fmt_pm(r['mean'], r['std']),
                        fmt(p),
                        stars(p),
                        '是' if is_better else '否'
                    ))
            lines.append('')

lines.append('## 总体统计结论')
lines.append('')
for key in sorted(summary_counts):
    item = summary_counts[key]
    method, ref = key.split('_vs_')
    lines.append('- {} vs {}：在 {}/{} 个数据集-时间-指标组合上更优，其中 {}/{} 个达到 p<0.05。'.format(
        DISPLAY_NAMES[method],
        DISPLAY_NAMES[ref],
        item['better'],
        item['total'],
        item['sig'],
        item['total']
    ))
lines.append('')

records = []
for dataset, times in structured.items():
    for time, metrics in times.items():
        for metric, row in metrics.items():
            base_method = row['temmeDWR_bank2048']
            for method in ['temmeDWR_fifo', 'temmeDWR_random', 'temmeDWR_random_reweight', 'latent_only_bank0']:
                cur = row[method]
                delta = cur['mean'] - base_method['mean']
                effect = -delta if metric in LOWER_BETTER else delta
                p = welch_pvalue_from_samples(cur['values'], base_method['values'])
                records.append((effect, dataset, time, metric, method, delta, p))

records.sort(reverse=True)
lines.append('## 相对 temmeDWR(bank2048) 的代表性结果')
lines.append('')
lines.append('### 提升最明显的 12 项')
lines.append('')
lines.append('| 数据集 | 时间 | 指标 | 方法 | 相对 bank2048 改变量 | p 值 | 显著性 |')
lines.append('|---|---:|---|---|---:|---:|:---:|')
for effect, dataset, time, metric, method, delta, p in records[:12]:
    lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
        dataset, time, metric, DISPLAY_NAMES[method], fmt(delta), fmt(p), stars(p)
    ))
lines.append('')
lines.append('### 下降最明显的 12 项')
lines.append('')
lines.append('| 数据集 | 时间 | 指标 | 方法 | 相对 bank2048 改变量 | p 值 | 显著性 |')
lines.append('|---|---:|---|---|---:|---:|:---:|')
for effect, dataset, time, metric, method, delta, p in records[-12:]:
    lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
        dataset, time, metric, DISPLAY_NAMES[method], fmt(delta), fmt(p), stars(p)
    ))
lines.append('')

OUTPUT.write_text('\n'.join(lines), encoding='utf-8')
print(str(OUTPUT))
