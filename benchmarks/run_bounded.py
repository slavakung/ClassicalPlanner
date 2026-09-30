#!/usr/bin/env python3
"""Run an explicit list of benchmark cases once, with enforced process limits.

Only completed, replay-validated solutions and bounded no-plan results qualify.
Timeouts are not evidence that a case fits in memory. Requires POSIX resource
limits; Linux additionally provides sampled process virtual-memory high water.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STOP = False
ALGORITHMS = set(('bfs dfs recursive-dfs bbdfs ucs astar greedy weighted-astar beam '
                  'iddfs idastar rbfs hill ehc backward-bfs bidirectional forward-vi '
                  'backward-vi rollout adp adp-astar graphplan satplan regression csp partial-order').split())


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def records(path):
    """Repair an incomplete final write while holding the exclusive output lock."""
    if not path.exists():
        return []
    result = []
    with path.open('rb+') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('result record must be an object')
            except json.JSONDecodeError:
                if stream.read():
                    raise ValueError('corrupt non-final result record')
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                break
            result.append(row)
            if not line.endswith(b'\n'):
                stream.write(b'\n')
                stream.flush()
                os.fsync(stream.fileno())
    return result


def stop(signum, frame):
    global STOP
    STOP = True


def child_limits(memory_bytes):
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.nice(10)


def terminate(child):
    """Kill the whole session, including descendants after its leader exits."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def memory_peak(pid):
    values = {}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmHWM', 'VmPeak'):
                values[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    return values.get('VmHWM'), values.get('VmPeak')


def maximum(previous, current):
    if current is None:
        return previous
    return max(previous or 0, current)


def classify(record, output, returncode):
    """Require a matching exit status and independent replay before qualifying."""
    try:
        lines = [line for line in output.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError('expected one benchmark JSON record')
        measurement = json.loads(lines[0], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(measurement, dict):
            raise ValueError('benchmark JSON must be an object')
        record['measurement'] = measurement
        reported = measurement.get('status')
        if (reported, returncode) not in (('solved', 0), ('unsolved', 2), ('error', 1), ('resource_exhausted', 3)):
            raise ValueError('benchmark status does not match exit code')
        record['status'] = reported
        if reported == 'solved' and measurement.get('replay_valid') is not True:
            record['status'] = 'invalid_plan'
        elif reported == 'error' and measurement.get('phase') == 'replay':
            record['status'] = 'invalid_plan'
        measured_rss = measurement.get('peak_rss_kib')
        if measured_rss is not None:
            if isinstance(measured_rss, bool) or not isinstance(measured_rss, (int, float)) or not math.isfinite(measured_rss) or measured_rss < 0:
                raise ValueError('invalid peak_rss_kib measurement')
            record['peak_rss_kib'] = maximum(record['peak_rss_kib'], measured_rss)
        for field in ('algorithm', 'max_depth'):
            if field in measurement and measurement[field] != record[field]:
                raise ValueError(f'benchmark returned a different {field}')
    except (ValueError, TypeError) as exc:
        record['status'] = 'error'
        record['error'] = str(exc)
    for field in ('peak_rss_kib', 'peak_virtual_kib'):
        if record.get(field) is not None and record[field] * 1024 >= record['memory_limit_bytes']:
            if record['status'] in ('solved', 'unsolved'):
                record['status'] = 'resource_exhausted'
                record['error'] = 'observed process memory reached the configured limit'
    record['qualified'] = record['status'] in ('solved', 'unsolved')


def run_case(case, index, executable, inputs, output, spec_hash, memory_bytes, default_timeout):
    instance = case['instance']
    timeout = case.get('timeout_seconds', default_timeout)
    command = [str(executable), '--domain', str(inputs[instance['domain']]),
               '--problem', str(inputs[instance['problem']]), '--algorithm', case['algorithm'],
               '--heuristic', 'hmax', '--max-depth', str(case['max_depth']), '--repetitions', '1']
    row = {'case_id': index, 'case_sha256': object_digest(case), 'instance_id': instance['id'],
           'algorithm': case['algorithm'], 'max_depth': case['max_depth'],
           'started_utc': utc(), 'timeout_seconds': timeout, 'memory_limit_bytes': memory_bytes,
           'memory_limit_kind': 'RLIMIT_AS', 'spec_sha256': spec_hash,
           'status': 'error', 'qualified': False, 'peak_rss_kib': None, 'peak_virtual_kib': None}
    started = time.monotonic()
    logs = output / 'jobs'
    logs.mkdir(exist_ok=True)
    child = None
    with (logs / f'{index:05d}.stdout').open('w+') as out, (logs / f'{index:05d}.stderr').open('w+') as err:
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=err, start_new_session=True,
                                     preexec_fn=lambda: child_limits(memory_bytes),
                                     env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            while child.poll() is None:
                rss, virtual = memory_peak(child.pid)
                row['peak_rss_kib'] = maximum(row['peak_rss_kib'], rss)
                row['peak_virtual_kib'] = maximum(row['peak_virtual_kib'], virtual)
                elapsed = time.monotonic() - started
                if STOP or elapsed >= timeout:
                    row['status'] = 'interrupted' if STOP else 'timeout'
                    terminate(child)
                    break
                try:
                    child.wait(timeout=min(.05, max(.001, timeout - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            row['returncode'] = child.returncode
            row['wall_seconds'] = time.monotonic() - started
            out.seek(0)
            data = out.read(8 * 1024 * 1024 + 1)
            if row['status'] == 'error':
                if row['wall_seconds'] > timeout:
                    row['status'] = 'timeout'
                elif len(data) > 8 * 1024 * 1024:
                    row['error'] = 'benchmark output exceeded 8 MiB'
                else:
                    classify(row, data, child.returncode)
        except (OSError, subprocess.SubprocessError) as exc:
            row['error'] = str(exc)
        finally:
            if child is not None:
                terminate(child)
        err.seek(0, os.SEEK_END)
        err.seek(max(0, err.tell() - 3000))
        row['stderr_tail'] = err.read()
    row['wall_seconds'] = time.monotonic() - started
    row['finished_utc'] = utc()
    return row


def load_cases(path, default_timeout):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a JSON object')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest must contain a nonempty explicit cases list')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('each case must be an object')
        instance = case['instance']
        if not isinstance(instance, dict) or not all(isinstance(instance.get(k), str) and instance[k] for k in ('id', 'domain', 'problem')):
            raise ValueError('each instance needs id, domain and problem strings')
        if not isinstance(case['algorithm'], str) or case['algorithm'] not in ALGORITHMS:
            raise ValueError('algorithm must be a known planner name')
        if isinstance(case['max_depth'], bool) or not isinstance(case['max_depth'], int) or not 0 <= case['max_depth'] <= 2**64 - 1:
            raise ValueError('max_depth must be an unsigned 64-bit integer')
        timeout = case.get('timeout_seconds', default_timeout)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('case timeout must be positive and finite')
        identity = (instance['id'], case['algorithm'], case['max_depth'])
        if identity in seen:
            raise ValueError('duplicate instance/algorithm/depth case')
        seen.add(identity)
        for key in ('domain', 'problem'):
            relative = Path(instance[key])
            if relative.is_absolute() or not (ROOT / relative).resolve().is_relative_to(ROOT):
                raise ValueError('input paths must be relative files inside the source root')
    return manifest, cases


def enforce_public_allowlist(cases, timeout, memory):
    """A source bundle permits only its published pairs, inputs and resource caps."""
    public_path = ROOT / 'benchmarks/public_cases.json'
    if not public_path.exists():
        return
    public = json.loads(public_path.read_text())
    public_timeout = public.get('timeout_seconds', 120)
    _, allowed_cases = load_cases(public_path, public_timeout)
    allowed = {(c['instance']['id'], c['algorithm'], c['max_depth']): c for c in allowed_cases}
    if memory > public.get('memory_limit_bytes', 2 * 1024**3):
        raise ValueError('memory limit exceeds the public qualification limit')
    for case in cases:
        key = (case['instance']['id'], case['algorithm'], case['max_depth'])
        approved = allowed.get(key)
        if approved is None:
            raise ValueError('case is outside the published qualified case list')
        if case.get('timeout_seconds', timeout) > approved.get('timeout_seconds', public_timeout):
            raise ValueError('case timeout exceeds its public qualification limit')
        for field in ('domain', 'problem'):
            if case['instance'][field] != approved['instance'][field]:
                raise ValueError('case inputs differ from the published qualified case')
            expected = approved['instance'].get(field + '_sha256')
            if expected is not None and digest(ROOT / case['instance'][field]) != expected:
                raise ValueError('published case input digest does not match')


def main(argv=None):
    global STOP
    STOP = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/benchmark_planner')
    parser.add_argument('--cases', '--manifest', dest='cases', type=Path, default=ROOT / 'benchmarks/public_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results')
    parser.add_argument('--timeout-seconds', type=float)
    parser.add_argument('--memory-gib', type=float)
    args = parser.parse_args(argv)
    try:
        preliminary = json.loads(args.cases.read_text())
        if not isinstance(preliminary, dict):
            raise ValueError('manifest must be a JSON object')
        timeout = args.timeout_seconds if args.timeout_seconds is not None else preliminary.get('timeout_seconds', 120)
        memory = int(args.memory_gib * 1024**3) if args.memory_gib is not None else preliminary.get('memory_limit_bytes', 2 * 1024**3)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise ValueError('memory limit must be a positive byte count')
        manifest, cases = load_cases(args.cases, timeout)
        enforce_public_allowlist(cases, timeout, memory)
        args.binary = args.binary.resolve(strict=True)
        if not os.access(args.binary, os.X_OK):
            raise ValueError('binary must be executable')
        inputs = {case['instance'][key]: digest(ROOT / case['instance'][key])
                  for case in cases for key in ('domain', 'problem')}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'qualification.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another runner owns this output directory')
        spec = {'schema_version': 1, 'binary_sha256': digest(args.binary),
                'runner_sha256': digest(__file__), 'inputs_sha256': inputs, 'cases': cases,
                'timeout_seconds': timeout, 'memory_limit_bytes': memory,
                'heuristic': 'hmax', 'repetitions': 1, 'nice_adjustment': 10,
                'memory_limit_kind': 'RLIMIT_AS', 'poll_seconds': .05}
        spec_hash = object_digest(spec)
        metadata_path = args.output / 'qualification.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata['spec_sha256'] != spec_hash:
                parser.error('binary, inputs, runner or protocol changed; use a new output directory')
        else:
            metadata = {'schema_version': 1, 'started_utc': utc(), 'spec_sha256': spec_hash, 'spec': spec,
                        'environment': {'platform': platform.platform(), 'python': sys.version},
                        'note': 'One fresh serial process per explicit case; capped bounded no-plan is not proof of unsolvability. Timeouts do not qualify.'}
            atomic(metadata_path, metadata)
        frozen = args.output / 'frozen'
        frozen.mkdir(exist_ok=True)
        def snapshot(source, expected):
            target = frozen / expected
            if not target.exists():
                temporary = frozen / (expected + '.tmp')
                shutil.copy2(source, temporary)
                if digest(temporary) != expected:
                    raise ValueError('input changed while freezing qualification')
                temporary.replace(target)
            if digest(target) != expected:
                raise ValueError('frozen qualification input was modified')
            return target.resolve()
        executable = snapshot(args.binary, spec['binary_sha256'])
        frozen_inputs = {name: snapshot(ROOT / name, value) for name, value in inputs.items()}
        raw = args.output / 'raw.jsonl'
        prior = records(raw)
        for row in prior:
            index = row.get('case_id')
            if row.get('spec_sha256') != spec_hash or not isinstance(index, int) or not 0 <= index < len(cases) or row.get('case_sha256') != object_digest(cases[index]):
                parser.error('result record does not match this qualification specification')
        completed = {row['case_id']: row for row in prior if row['status'] != 'interrupted'}
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        def heartbeat(state, active=None):
            atomic(args.output / 'status.json', {'schema_version': 1, 'pid': os.getpid(), 'status': state,
                'updated_utc': utc(), 'spec_sha256': spec_hash, 'total_cases': len(cases),
                'completed_cases': len(completed), 'qualified_cases': sum(row['qualified'] for row in completed.values()),
                'status_counts': dict(Counter(row['status'] for row in completed.values())),
                'active_case_id': active})
        heartbeat('running')
        for index, case in enumerate(cases):
            if STOP:
                break
            if index in completed:
                continue
            heartbeat('running', index)
            row = run_case(case, index, executable, frozen_inputs, args.output, spec_hash, memory, timeout)
            with raw.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if row['status'] != 'interrupted':
                completed[index] = row
            heartbeat('running')
            print(json.dumps({'completed': len(completed), 'total': len(cases), 'instance_id': row['instance_id'],
                              'algorithm': row['algorithm'], 'status': row['status'], 'qualified': row['qualified']}), flush=True)
        heartbeat('interrupted' if STOP else 'complete')
    return 130 if STOP else 0


if __name__ == '__main__':
    raise SystemExit(main())
