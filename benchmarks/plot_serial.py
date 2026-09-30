#!/usr/bin/env python3
"""Generate portable SVG/PDF/PNG comparisons, retaining failures and censoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import statistics
from measurement_context import annotate

os.environ.setdefault('MPLCONFIGDIR', '/tmp/planning-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

COLORS = {'solved': '#137c8b', 'timeout': '#e8a54b', 'unsolved': '#a6bac8',
          'resource_exhausted': '#9273b8', 'error': '#d46666', 'invalid_plan': '#922d50', 'interrupted': '#8b83ae', 'deadline': '#8b83ae'}


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = []
    lines = args.input.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    # A resumed job supersedes its interrupted attempt. No duplicate weighting.
    rows = list({row['job_id']: row for row in raw}.values())
    context_path = args.input.parent / 'measurement_context.json'
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    rows = annotate(rows, context)
    if not rows:
        raise SystemExit('no results to plot')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#344354',
                         'text.color': '#233347', 'axes.edgecolor': '#ccd3db',
                         'figure.facecolor': 'white', 'savefig.dpi': 180,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    algorithms = sorted({r['algorithm'] for r in rows})
    instances = sorted({r['instance_id'] for r in rows}, key=lambda s: (next(r['family'] for r in rows if r['instance_id'] == s), s))
    groups = defaultdict(list)
    for row in rows:
        groups[row['algorithm'], row['instance_id']].append(row)
    aggregate = []
    for (algorithm, instance), values in sorted(groups.items()):
        solved = [v for v in values if v['status'] == 'solved' and v.get('measurement', {}).get('replay_valid') is True]
        measured = [v['measurement'] for v in solved]
        timed = [v['measurement'] for v in solved if v['timing_eligible']]
        item = {'algorithm': algorithm, 'instance_id': instance, 'family': values[0]['family'],
                'attempts': len(values), 'solved': len(solved), 'status_counts': dict(Counter(v['status'] for v in values)),
                'timing_samples': len(timed), 'excluded_timing_samples': len(measured) - len(timed),
                'reference_cost': values[0].get('reference_cost')}
        for key in ('planning_seconds', 'parse_seconds', 'grounding_seconds', 'plan_cost', 'plan_length', 'peak_rss_kib'):
            source = timed if key.endswith('_seconds') else measured
            series = [v[key] for v in source if number(v.get(key))]
            item[key] = statistics.median(series) if series else None
        walls = [v['wall_seconds'] for v in solved if v['timing_eligible'] and number(v.get('wall_seconds'))]
        item['wall_seconds'] = statistics.median(walls) if walls else None
        aggregate.append(item)
    excluded = sum(not row['timing_eligible'] for row in rows)
    (args.output / 'summary.json').write_text(json.dumps({'attempts': len(rows), 'timing_excluded_attempts': excluded,
                                                       'measurement_context': context, 'results': aggregate}, indent=2) + '\n')

    def save(fig, stem, note):
        fig.text(.01, .012, note, fontsize=8, color='#607083', ha='left', va='bottom')
        fig.tight_layout(rect=(0, .05, 1, 1))
        for extension in ('svg', 'pdf', 'png'):
            target = args.output / f'{stem}.{extension}'
            fig.savefig(target, bbox_inches='tight')
            if extension == 'svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, max(4, .30 * len(algorithms) + 1.6)))
    left = np.zeros(len(algorithms))
    statuses = sorted({r['status'] for r in rows}, key=lambda s: list(COLORS).index(s) if s in COLORS else 99)
    for status in statuses:
        values = [sum(r['algorithm'] == a and r['status'] == status for r in rows) for a in algorithms]
        ax.barh(algorithms, values, left=left, color=COLORS.get(status, '#888888'), label=status.replace('_', ' '), height=.7)
        left += values
    ax.invert_yaxis()
    ax.set(xlabel='Measured attempts (including repeated rounds)', title='Serial planning • outcomes')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=min(4, len(statuses)), frameon=False)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    save(fig, 'outcomes', 'Unattempted jobs are absent. Bounded no-plan results are not general unsolvability proofs.')

    matrix = np.full((len(algorithms), len(instances)), np.nan)
    for item in aggregate:
        if number(item['planning_seconds']) and item['planning_seconds'] > 0:
            matrix[algorithms.index(item['algorithm']), instances.index(item['instance_id'])] = item['planning_seconds']
    fig, ax = plt.subplots(figsize=(max(10, .29 * len(instances) + 3), max(4, .30 * len(algorithms) + 2.5)))
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        palette = plt.get_cmap('YlGnBu').copy()
        palette.set_bad('#e7ebef')
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=palette,
                       norm=LogNorm(vmin=max(low, 1e-8), vmax=max(high, low * 1.01)))
        fig.colorbar(im, ax=ax, label='Median search time, seconds (log scale)', shrink=.8, pad=.02)
    else:
        ax.imshow(np.zeros_like(matrix), aspect='auto', cmap='Greys', vmin=0, vmax=1)
        ax.text(.5, .5, 'No eligible timing samples yet', transform=ax.transAxes,
                ha='center', va='center', color='#344354', bbox={'facecolor': 'white', 'edgecolor': 'none'})
    ax.set_xticks(range(len(instances)), instances, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(algorithms)), algorithms)
    ax.set_title('Serial planning • search time by task', pad=16)
    save(fig, 'search_time', 'Replayed solutions with eligible timing only. Grey = no eligible timing; excluded/timeout runs remain in outcomes.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(1, len(algorithms))))
    for a, color in zip(algorithms, colors):
        solved = sorted(item['wall_seconds'] for item in aggregate if item['algorithm'] == a and number(item['wall_seconds']) and item['wall_seconds'] > 0)
        if solved:
            axes[0].step(solved, range(1, len(solved) + 1), where='post', label=a, color=color, linewidth=1.6)
    axes[0].set(xscale='log', xlabel='Median process wall time, seconds', ylabel='Distinct tasks solved', title='Solved-task runtime curves')
    axes[0].grid(alpha=.18)
    ratios, labels = [], []
    for a in algorithms:
        values = [item['plan_cost'] / item['reference_cost'] for item in aggregate if item['algorithm'] == a and number(item['plan_cost']) and number(item['reference_cost']) and item['reference_cost'] > 0]
        if values:
            ratios.append(values)
            labels.append(a)
    if ratios:
        axes[1].boxplot(ratios, vert=False, labels=labels, patch_artist=True,
                        boxprops={'facecolor': '#b8dce0', 'edgecolor': '#137c8b'}, medianprops={'color': '#d07c31'})
        axes[1].axvline(1, linestyle='--', linewidth=1, color='#637788')
    axes[1].set(xlabel='Returned cost / known reference cost', title='Solution quality on solved tasks')
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=8 if len(algorithms) <= 10 else 6,
                       ncol=1 if len(algorithms) <= 10 else 2, loc='best')
    else:
        axes[0].text(.5, .5, 'No eligible timing samples yet', transform=axes[0].transAxes,
                     ha='center', va='center', color='#344354')
    save(fig, 'runtime_quality', 'Curves use solved tasks only; compare with outcomes and task coverage. Quality ratios use generated-task reference costs.')

    report = ['# Serial comparison results', '', f'{len(rows)} recorded attempts; {len(instances)} tasks; {len(algorithms)} algorithms.', '',
              'These are local measurements, not a completed seven-day study unless the campaign status confirms it.', '',
              '![Outcomes](outcomes.svg)', '', '![Search times](search_time.svg)', '', '![Runtime and quality](runtime_quality.svg)', '',
              'Each solved plan passed independent replay. Search time excludes input parsing and grounding; wall time includes process startup.',
              'Timeouts and missing observations are not converted to slow solved runs. Repeated rounds are not independent machines.', '',
              f'{excluded} attempts excluded from timing summaries by execution context; their measured outcomes and solution costs remain included.', '',
              '| Algorithm | Attempts | Solved attempts | Distinct tasks solved |', '|---|---:|---:|---:|']
    for a in algorithms:
        selected = [r for r in rows if r['algorithm'] == a]
        report.append(f"| {a} | {len(selected)} | {sum(r['status'] == 'solved' for r in selected)} | {len({r['instance_id'] for r in selected if r['status'] == 'solved'})} |")
    (args.output / 'README.md').write_text('\n'.join(report) + '\n')
    print(f'Wrote 3 figures (SVG/PDF/PNG), summary.json and README.md to {args.output}')


if __name__ == '__main__':
    main()
