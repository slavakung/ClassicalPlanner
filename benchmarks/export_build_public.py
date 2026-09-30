#!/usr/bin/env python3
"""Export source, all tests, and an explicitly memory-qualified benchmark suite.

The qualification record stays outside the portable bundle. Its digest and the
selected bounded cases provide provenance without redistributing campaign logs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from public_results import ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'classical-planner-build-public-source'
MANIFEST = 'BUILD_PUBLIC_MANIFEST.json'
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3
ROOT_NAMES = {'CMakeLists.txt', 'LICENSE', 'NOTICE', 'PDDL_SUPPORTED.md'}
DIRECTORY_SUFFIXES = {
    'cmake': {'.cmake'}, 'pddl': {'.h', '.hpp', '.cpp'},
    'ir': {'.h', '.hpp', '.cpp'}, 'symbolic': {'.h', '.hpp', '.cpp'},
    'tests': {'.h', '.hpp', '.cpp', '.py'}, 'examples': {'.pddl'},
}
BENCHMARK_NAMES = {
    'benchmark_planner.cpp', 'verify_pairs.cpp', 'generate_instances.py',
    'manifest.json', 'serial_campaign.py', 'run_bounded.py',
    'measurement_context.py', 'public_results.py', 'export_public.py',
    'export_build_public.py', 'plot_public.py', 'plot_serial.py',
    'export_results.py', 'requirements.txt',
}
EXCLUDED_PARTS = {'results', 'public', 'frozen', 'bin', '__pycache__', '.git', '.venv'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_file(name):
    """Only canonical, repository-relative POSIX file paths are accepted."""
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid relative source path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts or str(path) != name:
        raise ValueError('invalid relative source path: ' + name)
    return path


def checked_file(repository, path):
    relative = path.relative_to(repository)
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != repository.parent):
        raise ValueError('refusing source symlink: ' + str(relative))
    if not path.is_file() or not path.resolve().is_relative_to(repository):
        raise ValueError('source is not a regular file: ' + str(relative))
    return path


def ignored(parts):
    return any(part in EXCLUDED_PARTS or part.startswith('build') for part in parts)


def source_files(repository=ROOT):
    """An allowlist prevents logs, binaries and historical snapshots escaping."""
    repository = Path(repository).resolve()
    paths = []
    for path in repository.iterdir():
        if path.suffix in {'.h', '.hpp', '.cpp'} or path.name in ROOT_NAMES:
            paths.append(checked_file(repository, path))
    for name, suffixes in DIRECTORY_SUFFIXES.items():
        folder = repository / name
        if folder.is_symlink():
            raise ValueError('refusing source symlink: ' + name)
        for path in folder.rglob('*'):
            if ignored(path.relative_to(folder).parts):
                continue
            if path.is_symlink():
                raise ValueError('refusing source symlink: ' + str(path.relative_to(repository)))
            if path.is_file() and path.suffix in suffixes:
                paths.append(checked_file(repository, path))
    for name in BENCHMARK_NAMES:
        path = repository / 'benchmarks' / name
        if path.exists() or path.is_symlink():
            paths.append(checked_file(repository, path))
    fixtures = repository / 'benchmarks/instances/generated'
    if fixtures.is_symlink():
        raise ValueError('refusing source symlink: benchmarks/instances/generated')
    for path in fixtures.rglob('*'):
        if path.is_symlink():
            raise ValueError('refusing source symlink: ' + str(path.relative_to(repository)))
        if path.is_file() and path.suffix == '.pddl':
            paths.append(checked_file(repository, path))
    return sorted(set(paths), key=lambda path: str(path.relative_to(repository)))


def qualification_source_hashes(repository=ROOT):
    """Hashes of every C++ build input; reporting/test Python may evolve separately."""
    repository = Path(repository).resolve()
    return {str(path.relative_to(repository)): digest(path.read_bytes())
            for path in source_files(repository)
            if path.suffix in {'.h', '.hpp', '.cpp', '.cmake'} or path.name == 'CMakeLists.txt'}


def fixture_hashes(repository, manifest):
    result = {}
    for instance in manifest['instances']:
        for field in ('domain', 'problem'):
            name = str(relative_file(instance[field]))
            path = checked_file(repository, repository / name)
            if not name.startswith('benchmarks/instances/generated/') or path.suffix != '.pddl':
                raise ValueError('benchmark input must be a generated PDDL fixture')
            result[name] = digest(path.read_bytes())
            if instance.get(field + '_sha256') != result[name]:
                raise ValueError('fixture differs from benchmark manifest: ' + name)
    return result


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def curated_cases(qualification, repository=ROOT):
    repository = Path(repository).resolve()
    if qualification.get('schema_version') != 1:
        raise ValueError('unsupported qualification schema')
    limit = qualification.get('memory_limit_bytes')
    timeout = qualification.get('timeout_seconds')
    if limit != MEMORY_LIMIT_BYTES or not finite_positive(timeout):
        raise ValueError('qualification must use the 2 GiB limit and a positive timeout')
    manifest = json.loads((repository / 'benchmarks/manifest.json').read_text())
    if qualification.get('source_sha256') != qualification_source_hashes(repository):
        raise ValueError('qualification does not match current C++ build inputs')
    if qualification.get('inputs_sha256') != fixture_hashes(repository, manifest):
        raise ValueError('qualification does not match current benchmark inputs')
    instances = {instance['id']: instance for instance in manifest['instances']}
    selected, seen = [], set()
    rows = qualification.get('cases')
    if not isinstance(rows, list) or not rows:
        raise ValueError('qualification has no cases')
    for row in rows:
        key = (row.get('algorithm'), row.get('instance_id'))
        if key in seen:
            raise ValueError('duplicate qualification case: ' + repr(key))
        seen.add(key)
        if key[0] not in ALGORITHMS or key[1] not in instances:
            raise ValueError('unknown qualification case: ' + repr(key))
        if row.get('qualified') is not True:
            continue
        status = row.get('status')
        if status not in {'solved', 'unsolved', 'bounded_unsolved'}:
            raise ValueError('qualified benchmark must have completed without an error')
        if status == 'solved' and row.get('measurement', {}).get('replay_valid') is not True:
            raise ValueError('qualified solved benchmark must have an independently replayed plan')
        actual_limit = row.get('memory_limit_bytes', limit)
        actual_timeout = row.get('timeout_seconds', timeout)
        if (not finite_positive(actual_limit) or actual_limit > limit
                or not finite_positive(actual_timeout) or actual_timeout > timeout):
            raise ValueError('qualified case has inconsistent resource limits')
        for field in ('peak_rss_kib', 'peak_virtual_kib'):
            value = row.get(field)
            if field == 'peak_virtual_kib' and value is None:
                continue
            if not finite_positive(value) or value * 1024 >= actual_limit:
                raise ValueError('qualified case exceeds memory limit or lacks a peak measurement')
        depth = row.get('max_depth')
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError('qualified case must record its nonnegative search depth')
        selected.append({'algorithm': key[0], 'instance_id': key[1],
                         'instance': instances[key[1]], 'max_depth': depth,
                         'qualification': {
                             'status': status, 'memory_limit_bytes': actual_limit,
                             'timeout_seconds': actual_timeout, 'peak_rss_kib': row['peak_rss_kib'],
                             'peak_virtual_kib': row.get('peak_virtual_kib'),
                             'replay_valid': row.get('measurement', {}).get('replay_valid')
                         }})
    if not selected:
        raise ValueError('no completed memory-qualified benchmarks')
    return {'schema_version': 1, 'memory_limit_bytes': limit, 'timeout_seconds': timeout,
            'selection_rule': 'completed without errors below the memory limit; solved plans independently replayed',
            'cases': selected}, len(rows)


def overlap(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_destination(output, repository, evidence):
    if overlap(output, repository) or overlap(output, evidence):
        raise ValueError('output must not overlap source or qualification evidence')
    if not output.exists():
        return
    if output.is_symlink() or not output.is_dir():
        raise ValueError('output must be an exporter-owned directory')
    marker = output / MANIFEST
    if not marker.is_file() or marker.is_symlink():
        raise ValueError('existing output is not an exporter-owned directory')
    prior = json.loads(marker.read_text())
    if prior.get('owner') != OWNER or prior.get('schema_version') != 1 or not isinstance(prior.get('files'), dict):
        raise ValueError('existing output has no valid ownership manifest')
    inventory = prior['files']
    for name in inventory:
        relative_file(name)
    expected = set(inventory) | {MANIFEST}
    actual = set()
    expected_directories = {str(parent) for name in inventory for parent in PurePosixPath(name).parents if str(parent) != '.'}
    for path in output.rglob('*'):
        if path.is_symlink():
            raise ValueError('refusing to replace a modified output containing symlinks')
        if path.is_dir() and str(path.relative_to(output)) not in expected_directories:
            raise ValueError('refusing to replace an unowned output directory')
        if path.is_file():
            name = str(path.relative_to(output))
            actual.add(name)
            if name != MANIFEST and (name not in inventory or digest(path.read_bytes()) != inventory[name].get('sha256')):
                raise ValueError('refusing to replace modified or unowned output file: ' + name)
    if actual != expected:
        raise ValueError('refusing to replace an incomplete or modified output')


def readme(cases, evaluated, qualification):
    fresh = qualification.get('fresh_checked_cases')
    historical = qualification.get('historical_excluded_cases')
    coverage = ''
    if isinstance(fresh, int) and isinstance(historical, int):
        coverage = (f'Fresh qualification attempted {fresh} pairs; {historical} pairs were '
                    'excluded using earlier timeout evidence and were not rerun.\n\n')
    return f'''# Serial planner source and bounded benchmarks

This source-only distribution contains all 26 serial planning algorithms, all
C++ and Python tests, and **{len(cases['cases'])} selected algorithm/problem pairs**
from {evaluated} considered pairs. Selected runs completed without an error
under a **2 GiB virtual-address-space limit**, with measured peak memory below
that limit and a **{cases['timeout_seconds']:g}-second per-case timeout**. Solved
plans passed independent replay. Completion at a configured depth can also mean
bounded-unsolved; that outcome does not establish infeasibility.

{coverage}The selection is [benchmarks/public_cases.json](benchmarks/public_cases.json).
Each selected pair records its completion status, measured memory and applied
limits in that manifest. Checks may use a stricter 512 MiB limit; completion
under that cap also qualifies below 2 GiB. All 48 generated input fixtures remain
available for regression and generator tests; only the explicit selected pairs
constitute this benchmark suite. Source
files, test support and fixture definitions are included. Compiled binaries,
historical results and live campaign files are excluded.

A C++23 compiler, CMake 3.24 or newer and Python 3.9 or newer are required.
Build and run the tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --parallel 2
python3 -m unittest discover -s tests -p 'test_*.py'
python3 benchmarks/generate_instances.py --check
```

Run the selected benchmarks on Linux with the recorded per-case limits:

```sh
python3 benchmarks/run_bounded.py --cases benchmarks/public_cases.json --output benchmark-results
```

The runner defaults to `build/benchmark_planner` and runs explicit pairs only.
It reapplies the memory and time limits; performance and memory can vary with
the compiler, operating system and machine. Qualification describes the measured
configuration and is not a guarantee for every build. It creates fresh local
records in `benchmark-results`. Optional plotting support uses
`benchmarks/requirements.txt`; the build and bounded runner need no Python
packages. Legacy campaign/report modules remain as regression-test dependencies.

[BUILD_PUBLIC_MANIFEST.json](BUILD_PUBLIC_MANIFEST.json) records the qualification
evidence digest and SHA-256 inventory of every included file except itself.
Licensed under [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
'''


def export(qualification_path, output, repository=ROOT):
    repository = Path(repository).resolve()
    evidence_original = Path(qualification_path)
    output_original = Path(output)
    if evidence_original.is_symlink() or output_original.is_symlink():
        raise ValueError('qualification and output must not be symlinks')
    evidence, output = evidence_original.resolve(), output_original.resolve()
    validate_destination(output, repository, evidence)
    evidence_bytes = evidence.read_bytes()
    qualification = json.loads(evidence_bytes)
    cases, evaluated = curated_cases(qualification, repository)
    sources = source_files(repository)
    needed = {'CMakeLists.txt', 'LICENSE', 'NOTICE', 'tests.cpp',
              'benchmarks/run_bounded.py', 'benchmarks/export_build_public.py'}
    present = {str(path.relative_to(repository)) for path in sources}
    if needed - present:
        raise ValueError('missing required source files: ' + ', '.join(sorted(needed - present)))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.build-public-stage-', dir=output.parent))
    backup = None
    try:
        for path in sources:
            target = stage / path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (stage / 'benchmarks/public_cases.json').write_text(json.dumps(cases, indent=2, allow_nan=False) + '\n')
        (stage / 'README.md').write_text(readme(cases, evaluated, qualification))
        (stage / '.gitignore').write_text('build*/\nbenchmark-results/\n.venv/\n__pycache__/\n*.pyc\nbenchmarks/results/\n')
        inventory = {str(path.relative_to(stage)): {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
                     for path in sorted(stage.rglob('*')) if path.is_file()}
        record = {'schema_version': 1, 'owner': OWNER,
                  'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'qualification_sha256': digest(evidence_bytes),
                  'qualified_binary_sha256': qualification.get('binary_sha256'),
                  'memory_limit_bytes': cases['memory_limit_bytes'], 'timeout_seconds': cases['timeout_seconds'],
                  'algorithm_count': len(ALGORITHMS), 'qualified_case_count': len(cases['cases']),
                  'evaluated_case_count': evaluated,
                  'fresh_checked_cases': qualification.get('fresh_checked_cases'),
                  'historical_excluded_cases': qualification.get('historical_excluded_cases'), 'files': inventory}
        (stage / MANIFEST).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        # Refuse source edits made while staging: evidence must describe the copied build inputs.
        if qualification['source_sha256'] != qualification_source_hashes(stage):
            raise ValueError('source changed during export')
        if qualification['inputs_sha256'] != fixture_hashes(stage, json.loads((stage / 'benchmarks/manifest.json').read_text())):
            raise ValueError('benchmark inputs changed during export')
        validate_destination(output, repository, evidence)
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix='.build-public-old-', dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return record
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=ROOT)
    args = parser.parse_args()
    result = export(args.qualification, args.output, args.repository)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
