#!/usr/bin/env python3
"""Stage a portable Apache-2.0 source tree and complete, honestly labelled results."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from public_results import ALGORITHMS, build_matrix, digest, read_snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ('cmake', 'pddl', 'ir', 'symbolic', 'tests', 'examples', 'scripts', 'benchmarks', 'docs')
SOURCE_SUFFIXES = {'.h', '.hpp', '.cpp', '.cmake', '.py', '.sh', '.md', '.txt', '.json', '.pddl'}
EXCLUDED_COMPONENTS = {'results', 'build', '__pycache__', '.git', '.venv', 'frozen', 'bin', 'public'}


def normalized(value, repository=ROOT):
    """Keep hashes and scientific fields; redact machine-local path prefixes."""
    if isinstance(value, dict):
        return {key: normalized(item, repository) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item, repository) for item in value]
    if isinstance(value, str):
        value = value.replace(str(repository), '<repository>')
        value = re.sub(r'/home/[^/\s"\']+', '<home>', value)
        value = re.sub(r'/Users/[^/\s"\']+', '<home>', value)
        value = re.sub(r'/run/user/\d+', '<runtime-user>', value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_files(repository):
    repository = Path(repository)
    paths = [path for path in repository.iterdir()
             if path.is_file() and (path.suffix in ('.h', '.hpp', '.cpp')
                                   or path.name in ('CMakeLists.txt', 'LICENSE', 'NOTICE', 'PDDL_SUPPORTED.md', 'SOURCE_SNAPSHOT.json'))]
    for directory in SOURCE_DIRECTORIES:
        folder = repository / directory
        if not folder.exists():
            continue
        for path in folder.rglob('*'):
            relative = path.relative_to(folder)
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and not set(relative.parts) & EXCLUDED_COMPONENTS:
                if path.is_symlink():
                    raise ValueError('refusing symlink in source snapshot: ' + str(path.relative_to(repository)))
                paths.append(path)
    return sorted(set(paths), key=lambda path: str(path.relative_to(repository)))


def write_csv(path, matrix):
    fields = ['algorithm', 'instance_id', 'family', 'representation', 'outcome', 'active_job',
              'recorded_repetitions', 'status_counts', 'feasibility', 'infeasibility', 'optimality',
              'valid_replayed_solutions', 'reference_matching_solutions', 'best_plan_cost',
              'reference_optimal_cost', 'eligible_timing_samples', 'excluded_solved_timing_samples',
              'search_median_seconds', 'search_min_seconds', 'search_max_seconds', 'wall_median_seconds',
              'parse_median_seconds', 'grounding_median_seconds', 'replay_median_seconds',
              'timing_exclusion_reasons', 'configured_timeout_seconds', 'configured_memory_gib']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in matrix['cases']:
            item = {key: row.get(key) for key in fields}
            item['status_counts'] = json.dumps(row['status_counts'], sort_keys=True)
            item['timing_exclusion_reasons'] = '; '.join(row['timing_exclusion_reasons'])
            item['reference_optimal_cost'] = row['reference_evidence']['optimal_cost']
            for key, field in [('search', 'planning_seconds'), ('wall', 'wall_seconds'), ('parse', 'parse_seconds'),
                               ('grounding', 'grounding_seconds'), ('replay', 'replay_seconds')]:
                item[key + '_median_seconds'] = row['timing'][field]['median']
            item['search_min_seconds'] = row['timing']['planning_seconds']['min']
            item['search_max_seconds'] = row['timing']['planning_seconds']['max']
            writer.writerow(item)


def result_readme(matrix, figures):
    summary = matrix['summary']
    label = 'Final campaign snapshot' if summary['campaign_window_complete'] else 'PRELIMINARY — campaign window not completed'
    text = ['# ' + label, '',
            f"Captured: {summary['snapshot']['captured_utc']}. Original deadline: {summary['deadline_utc']}.", '',
            f"The full matrix contains **{summary['matrix_cases']} cases: {summary['algorithm_count']} algorithms × {summary['instance_count']} instances**. "
            f"{summary['attempted_cases']} cases have recorded attempts; {summary['cases_with_valid_solution']} have a replay-valid full solution; "
            f"{summary['cases_with_eligible_timing']} have eligible solved timing samples.", '',
            'Use [matrix.csv](matrix.csv) for every algorithm/problem pair and [matrix.json](matrix.json) for per-outcome counts, '
            'timing ranges, proof reasons and reference provenance. [raw.jsonl](raw.jsonl) retains all attempts, including superseded interruptions.', '',
            'Missing cells are explicitly `not_attempted` or `pending`. `unsolved` is exported as `bounded_unsolved`; '
            'it does not establish infeasibility. `resource_exhausted` records allocation failure, not a proof of infeasibility. '
            'Feasibility is proven only by a complete plan with recorded independent original grounded-IR replay. '
            'Optimality is reported only for unit-cost solutions matching a documented generated-task optimum on matching input hashes. '
            'Algorithm names alone are not certificates. Some repetitions can be optimal while others return higher costs.', '',
            'Timing summaries include only replay-valid solved samples that remain eligible under the captured '
            '[measurement_context.json](measurement_context.json). Excluded samples remain usable for correctness and solution-quality evidence. '
            'Timeouts and failures are retained, with no invented solved runtime. Medians and min/max ranges are descriptive; repeated runs share one host.', '']
    if figures:
        text += ['![Full case coverage](figures/coverage.svg)', '', '![Eligible solve times](figures/eligible_search_time.svg)', '',
                 '![Solution evidence](figures/solution_evidence.svg)', '',
                 'All three figures are available as SVG, PNG and PDF. Grey timing cells have no eligible solved sample; consult coverage and counts before comparing algorithms.', '']
    text += ['| Algorithm | Cases attempted | Cases feasible | Cases reaching reference optimum | Cases with eligible timing |',
             '|---|---:|---:|---:|---:|']
    for algorithm in matrix['algorithms']:
        selected = [row for row in matrix['cases'] if row['algorithm'] == algorithm]
        text.append(f"| {algorithm} | {sum(row['attempted'] for row in selected)} | "
                    f"{sum(row['valid_replayed_solutions'] > 0 for row in selected)} | "
                    f"{sum(row['reference_matching_solutions'] > 0 for row in selected)} | "
                    f"{sum(row['eligible_timing_samples'] > 0 for row in selected)} |")
    text += ['', 'The source snapshot and reproduction instructions are in the parent directory. '
             'Source JSON hashes describe captured bytes before presentation-only path normalization. '
             'The source campaign, its measurement context and its running processes were not modified by this export.', '']
    return '\n'.join(text)


def publication_readme(matrix):
    summary = matrix['summary']
    state = 'Final measurement snapshot' if summary['campaign_window_complete'] else 'PRELIMINARY — the seven-day campaign is still incomplete'
    return f'''# Public serial planning source and results

**{state}.** Snapshot: {summary['snapshot']['captured_utc']}.
The original campaign deadline is **{summary['deadline_utc']}**. This staged
folder is prepared for review and is not automatically published anywhere.

This source-only C++23 distribution contains all 26 serial algorithms, all 48
generated problems, the benchmark/generator, regression tests, campaign scripts,
and reporting tools. The [results](results/README.md) include the full 1,248-case
matrix, raw attempts, eligible timing summaries, feasibility/optimality evidence,
and portable figures. Missing or unfinished measurements are stated explicitly.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --parallel 2
python3 -m unittest discover -s tests -p 'test_*.py'
python3 benchmarks/generate_instances.py --check
```

Use a compatible C++23 compiler and CMake 3.24 or newer. See
[algorithm contracts](docs/ALGORITHMS.md), [PDDL support](PDDL_SUPPORTED.md),
[benchmark methodology](docs/BENCHMARKS.md), and [campaign instructions](docs/CAMPAIGN.md).

To reproduce a new seven-day run on your machine:

```sh
python3 -m venv .venv
.venv/bin/pip install -r benchmarks/requirements.txt
python3 benchmarks/serial_campaign.py --dry-run
PLANNING_PYTHON="$PWD/.venv/bin/python" bash scripts/run_serial_week.sh
```

The runner defaults to the same 26×48 matrix, 120 seconds per attempt, 2 GiB
virtual address-space limit and seven calendar days. Record your CPU allocation,
software environment and overlapping work in a separate measurement context.
Fresh runs will not reproduce exact timing values across hosts; preserve the
published data when starting a new study.

Recreate the figures directly from the distributed matrix:

```sh
.venv/bin/python benchmarks/plot_public.py --input results/matrix.json --output results/figures
```

Refresh this source-only publication folder from a newly completed local run:

```sh
.venv/bin/python benchmarks/export_public.py --input benchmarks/results/week --output publication-refresh --require-complete
```

The [publication manifest](PUBLICATION_MANIFEST.json) records content hashes,
capture times and source-to-campaign provenance checks. The original campaign
recorded binary, runner and input hashes; it did not record a complete source
tree hash. The exporter therefore reports available checks explicitly and does
not claim that a source inventory alone cryptographically proves the original
binary build. No compiled binaries, SDKs, private transport configuration,
credentials, home-directory paths or live results directories are bundled.

Licensed under [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) and
[provenance](docs/PROVENANCE.md). All generated fixtures and their reference
derivations are included. No external IPC corpus is redistributed.
'''


def export(input_path, output, manifest_path=None, figures=True, require_complete=False, repository=ROOT):
    repository = Path(repository).resolve()
    source, output = Path(input_path).resolve(), Path(output).resolve()
    manifest_path = Path(manifest_path or repository / 'benchmarks/manifest.json')
    if output == repository or source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('publication output must not replace the repository or overlap live campaign files')
    if output.exists() and (not output.is_dir() or not (output / 'PUBLICATION_MANIFEST.json').exists()):
        raise ValueError('existing output is not an exporter-owned publication folder')
    snapshot = read_snapshot(source)
    manifest = json.loads(manifest_path.read_text())
    hashes = {instance[field]: digest((repository / instance[field]).read_bytes())
              for instance in manifest['instances'] for field in ('domain', 'problem')}
    matrix, rows = build_matrix(manifest, snapshot, hashes)
    if require_complete and not matrix['summary']['campaign_window_complete']:
        raise ValueError('campaign has not actually completed; refusing final publication export')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.public-stage-', dir=output.parent))
    try:
        source_inventory = {}
        for path in source_files(repository):
            relative = path.relative_to(repository)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_inventory[str(relative)] = digest(target.read_bytes())
        results = stage / 'results'
        results.mkdir()
        cases = snapshot['campaign'].get('spec', {}).get('cases', [])
        normalized_rows = []
        for row in rows:
            value = normalized(row, repository)
            try:
                instance = cases[int(row['job_id'].split(':')[1])]['instance']
                if 'measurement' in value:
                    value['measurement']['domain_file'] = instance['domain']
                    value['measurement']['problem_file'] = instance['problem']
            except (ValueError, IndexError, KeyError):
                pass
            normalized_rows.append(value)
        with (results / 'raw.jsonl').open('w') as stream:
            for row in normalized_rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        matrix = normalized(matrix, repository)
        write_json(results / 'matrix.json', matrix)
        write_json(results / 'summary.json', matrix['summary'])
        write_json(results / 'campaign.json', normalized(snapshot['campaign'], repository))
        write_json(results / 'source_status.json', normalized(snapshot['status'], repository))
        write_json(results / 'measurement_context.json', normalized(snapshot['context'], repository))
        write_json(results / 'benchmark_manifest.json', normalized(manifest, repository))
        write_csv(results / 'matrix.csv', matrix)
        if figures:
            from plot_public import make_figures
            make_figures(matrix, results / 'figures')
        (results / 'README.md').write_text(result_readme(matrix, figures))
        (stage / 'README.md').write_text(publication_readme(matrix))
        (stage / '.gitignore').write_text('build*/\n.venv/\n__pycache__/\n*.pyc\nbenchmarks/results/\n.public-stage-*/\n')
        spec = snapshot['campaign'].get('spec', {})
        current_binary = repository / 'build/benchmark_planner'
        runner = repository / 'benchmarks/serial_campaign.py'
        checks = {'campaign_binary_sha256': spec.get('binary_sha256'),
                  'current_build_matches_campaign_binary': digest(current_binary.read_bytes()) == spec.get('binary_sha256') if current_binary.exists() else None,
                  'current_runner_matches_campaign_runner': digest(runner.read_bytes()) == spec.get('runner_sha256') if runner.exists() else None,
                  'all_distributed_inputs_match_campaign': all(spec.get('inputs_sha256', {}).get(name) == value for name, value in hashes.items()),
                  'original_full_source_tree_hash_available': False,
                  'note': 'Current source inventory is an export-time snapshot. Original campaign has no full source-tree build hash.'}
        inventory = {str(path.relative_to(stage)): {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
                     for path in sorted(stage.rglob('*')) if path.is_file()}
        publication = {'schema_version': 1, 'publication_state': matrix['summary']['publication_state'],
                       'captured_utc': snapshot['snapshot']['captured_utc'], 'campaign_deadline_utc': matrix['summary']['deadline_utc'],
                       'campaign_started_utc': matrix['summary']['started_utc'],
                       'campaign_finished': matrix['summary']['campaign_finished'],
                       'campaign_window_complete': matrix['summary']['campaign_window_complete'],
                       'campaign_status': matrix['summary']['campaign_status'],
                       'configured_duration_days': matrix['summary']['configured_duration_days'],
                       'algorithm_count': len(ALGORITHMS), 'instance_count': len(manifest['instances']),
                       'matrix_cases': len(matrix['cases']), 'source_files': source_inventory,
                       'files': inventory, 'campaign_snapshot': snapshot['snapshot'], 'provenance_checks': checks,
                       'license': 'Apache-2.0', 'contains_compiled_binaries_or_sdk': False,
                       'source_campaign_modified': False,
                       'path_normalization': 'machine-local repository/home/runtime prefixes removed from exported JSON; scientific values and hashes retained'}
        write_json(stage / 'PUBLICATION_MANIFEST.json', publication)
        backup = None
        if output.exists():
            backup = output.with_name(output.name + '.previous-' + stage.name.removeprefix('.public-stage-'))
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return publication
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'public')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/manifest.json')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    try:
        publication = export(args.input, args.output, args.manifest, not args.no_figures, args.require_complete)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({'output': str(args.output.resolve()), 'publication_state': publication['publication_state'],
                      'matrix_cases': publication['matrix_cases'], 'files': len(publication['files'])}, indent=2))


if __name__ == '__main__':
    main()
