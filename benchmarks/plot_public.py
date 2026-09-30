#!/usr/bin/env python3
"""Publication figures from the complete matrix; missing work stays visible."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/planning-public-matplotlib')


def make_figures(matrix, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm, BoundaryNorm
    from matplotlib.patches import Patch
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    algorithms = matrix['algorithms']
    instances = matrix['instances']
    rows = {(row['algorithm'], row['instance_id']): row for row in matrix['cases']}
    summary = matrix['summary']
    state = summary['publication_state'].upper()
    captured = summary['snapshot']['captured_utc']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'savefig.dpi': 180, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    labels = [instance['id'] for instance in instances]
    shape = (len(algorithms), len(instances))

    def axes_grid(title):
        fig, ax = plt.subplots(figsize=(max(14, .36 * len(instances) + 3), max(6, .31 * len(algorithms) + 3.7)))
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha='right', fontsize=8)
        ax.set_yticks(range(len(algorithms)), algorithms)
        ax.set_title(title + ' · ' + state, pad=18)
        for column in range(1, len(instances)):
            if instances[column]['family'] != instances[column - 1]['family']:
                ax.axvline(column - .5, color='white', linewidth=2)
        return fig, ax

    def save(fig, stem, note):
        fig.text(.01, .015, note + '\nSnapshot: ' + captured, fontsize=8, color='#475569', va='bottom')
        fig.tight_layout(rect=(0, .075, 1, 1))
        for suffix in ('svg', 'pdf', 'png'):
            path = output / (stem + '.' + suffix)
            fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    categories = [('No attempt', '#e2e8f0'), ('Pending / interrupted', '#f7d58b'),
                  ('Timeout', '#d89742'), ('Bounded no plan', '#aab9cd'),
                  ('Error / invalid / mixed', '#c65a6c'), ('Replayed solution', '#55a9b4'),
                  ('Optimum reference matched', '#176448'), ('Resource exhausted', '#9273b8')]
    coverage = np.zeros(shape)
    for y, algorithm in enumerate(algorithms):
        for x, instance in enumerate(instances):
            row = rows[algorithm, instance['id']]
            if row['reference_matching_solutions']:
                value = 6
            elif row['valid_replayed_solutions']:
                value = 5
            elif row['outcome'] == 'not_attempted':
                value = 0
            elif row['outcome'] in ('pending', 'interrupted', 'deadline'):
                value = 1
            elif row['outcome'] == 'timeout':
                value = 2
            elif row['outcome'] == 'bounded_unsolved':
                value = 3
            elif row['outcome'] == 'resource_exhausted':
                value = 7
            else:
                value = 4
            coverage[y, x] = value
    fig, ax = axes_grid('All algorithm–problem cases: observed outcomes')
    ax.set_title('All algorithm–problem cases: observed outcomes · ' + state, pad=58)
    palette = ListedColormap([color for _, color in categories])
    ax.imshow(coverage, aspect='auto', cmap=palette, norm=BoundaryNorm(np.arange(-.5, len(categories) + .5), len(categories)))
    ax.legend(handles=[Patch(facecolor=color, label=label) for label, color in categories],
              loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=4, fontsize=8, frameon=False)
    save(fig, 'coverage', 'Best verified evidence across recorded repetitions. A green cell does not mean every repetition was optimal.\nBounded no-plan, resource exhaustion, errors and timeouts are not proofs of infeasibility; full counts are in matrix.csv/json.')

    timing = np.full(shape, np.nan)
    for y, algorithm in enumerate(algorithms):
        for x, instance in enumerate(instances):
            row = rows[algorithm, instance['id']]
            value = row['timing']['planning_seconds']['median']
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                timing[y, x] = value
    fig, ax = axes_grid('Eligible median search time, seconds')
    finite = timing[np.isfinite(timing)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        palette = plt.get_cmap('YlGnBu').copy()
        palette.set_bad('#e2e8f0')
        image = ax.imshow(np.ma.masked_invalid(timing), aspect='auto', cmap=palette,
                          norm=LogNorm(vmin=max(lo, 1e-12), vmax=max(hi, lo * 1.01)))
        fig.colorbar(image, ax=ax, label='Median search seconds · logarithmic scale', shrink=.8, pad=.015)
    else:
        ax.imshow(np.zeros(shape), aspect='auto', cmap=ListedColormap(['#e2e8f0']))
        ax.text(.5, .5, 'No eligible solved timing samples at this snapshot', transform=ax.transAxes,
                ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'none'})
    save(fig, 'eligible_search_time', 'Only complete replay-valid solutions outside recorded exclusion intervals. Grey means no eligible solved timing.\nCosts and coverage differ across algorithms; this plot alone does not establish a speedup or equal-quality comparison.')

    fig, ax = plt.subplots(figsize=(12, max(6, .31 * len(algorithms) + 2)))
    optimal, other, unproven = [], [], []
    for algorithm in algorithms:
        selected = [row for row in matrix['cases'] if row['algorithm'] == algorithm]
        optimal.append(sum(row['reference_matching_solutions'] > 0 for row in selected))
        other.append(sum(row['valid_replayed_solutions'] > 0 and row['reference_matching_solutions'] == 0 for row in selected))
        unproven.append(sum(row['valid_replayed_solutions'] == 0 for row in selected))
    ax.barh(algorithms, optimal, color='#176448', label='Reference optimum reached in at least one repetition')
    ax.barh(algorithms, other, left=optimal, color='#55a9b4', label='Other replay-valid full solution')
    ax.barh(algorithms, unproven, left=np.array(optimal) + np.array(other), color='#e2e8f0', label='No verified solution recorded')
    ax.invert_yaxis()
    ax.set_xlim(0, len(instances))
    ax.set(xlabel='Distinct benchmark instances (each counted once)',
           title='Feasibility and solution-quality evidence · ' + state)
    ax.set_title('Feasibility and solution-quality evidence · ' + state, pad=65)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=8)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    save(fig, 'solution_evidence', 'Every algorithm has the same full problem denominator. Reference matches require unit costs and matching input hashes.\nNo verified solution means not proven by the recorded runs, not that the task is infeasible.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    make_figures(json.loads(args.input.read_text()), args.output)


if __name__ == '__main__':
    main()
