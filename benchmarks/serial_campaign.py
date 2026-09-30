#!/usr/bin/env python3
"""Restartable, bounded serial comparison campaign; no plotting dependency to run."""
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
import random
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ('bfs dfs recursive-dfs bbdfs ucs astar greedy weighted-astar beam '
              'iddfs idastar rbfs hill ehc backward-bfs bidirectional forward-vi '
              'backward-vi rollout adp adp-astar graphplan satplan regression csp partial-order').split()
STOP = False


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def records(path):
    """Repair only an incomplete final write, under the exclusive campaign lock."""
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
            except json.JSONDecodeError:
                if stream.read():
                    raise ValueError('corrupt non-final result record')
                stream.truncate(offset)
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


def terminate(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        except ProcessLookupError:
            child.wait()


def refresh_figures(output):
    # A plotting failure must not lose measurements or stop the campaign.
    with (output / 'plot.log').open('a') as log:
        try:
            subprocess.run([sys.executable, str(ROOT / 'benchmarks/plot_serial.py'),
                            '--input', str(output / 'raw.jsonl'),
                            '--output', str(output / 'figures')], cwd=ROOT,
                           stdout=log, stderr=log, timeout=90, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.write(str(exc) + '\n')


def read_cases(path, instances):
    """Resolve explicit pairs against the fixture manifest without expanding them."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise ValueError('case manifest must be a schema_version 1 object')
    entries = document.get('cases')
    if not isinstance(entries, list) or not entries:
        raise ValueError('case manifest must contain a nonempty cases list')
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('each case must be an object')
        embedded = entry.get('instance')
        if embedded is not None and not isinstance(embedded, dict):
            raise ValueError('case instance metadata must be an object')
        instance_id = entry.get('instance_id', embedded.get('id') if embedded else None)
        if not isinstance(instance_id, str) or instance_id not in instances:
            raise ValueError('case contains an unknown instance ID')
        instance = instances[instance_id]
        if embedded is not None and embedded != instance:
            raise ValueError('case instance metadata does not match the fixture manifest')
        algorithm, horizon = entry.get('algorithm'), entry.get('max_depth')
        if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
            raise ValueError('case contains an unknown algorithm')
        if type(horizon) is not int or not 0 <= horizon <= 2**64 - 1:
            raise ValueError('case max_depth must be an unsigned 64-bit integer')
        pair = algorithm, instance_id
        if pair in seen:
            raise ValueError('case manifest contains a duplicate algorithm/instance pair')
        seen.add(pair)
        result.append({'instance': instance, 'algorithm': algorithm, 'max_depth': horizon})
    memory = document.get('memory_limit_bytes')
    if memory is not None and (type(memory) is not int or memory <= 0):
        raise ValueError('case memory_limit_bytes must be a positive integer')
    timeout = document.get('timeout_seconds')
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('case timeout_seconds must be positive and finite')
    return document, result


def case_key(case):
    return case['algorithm'], case['instance']['id'], case['max_depth']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/benchmark_planner')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/manifest.json')
    parser.add_argument('--cases', type=Path, help='explicit pair manifest; may only narrow bundled public_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/results/week')
    parser.add_argument('--duration-days', type=float, default=7)
    parser.add_argument('--timeout-seconds', type=float, help='default 120, capped by bundled case limits')
    parser.add_argument('--memory-gib', type=float, help='default 2, capped by bundled case limits')
    parser.add_argument('--algorithms', help='comma separated known algorithms; defaults to all permitted pairs')
    parser.add_argument('--instances', default='', help='comma separated manifest IDs; defaults to permitted tasks')
    parser.add_argument('--rounds', type=int, default=0, help='0 repeats full rounds until deadline')
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--plot-every', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.rounds < 0 or args.plot_every < 0:
        parser.error('rounds and plot-every must be nonnegative')
    args.binary = args.binary.resolve()
    algorithms = args.algorithms.split(',') if args.algorithms is not None else list(ALGORITHMS)
    if len(set(algorithms)) != len(algorithms) or not set(algorithms) <= set(ALGORITHMS):
        parser.error('algorithms must be unique known names')
    instance_ids = args.instances.split(',') if args.instances else []
    if len(set(instance_ids)) != len(instance_ids):
        parser.error('instance IDs must be unique')
    selected = set(instance_ids)
    case_sources, limits = {}, []
    bundled_path = ROOT / 'benchmarks/public_cases.json'
    try:
        instances = json.loads(args.manifest.read_text())['instances']
        by_id = {instance['id']: instance for instance in instances}
        if len(by_id) != len(instances):
            raise ValueError('fixture manifest contains duplicate instance IDs')
        if selected - by_id.keys():
            raise ValueError('instance selection contains unknown IDs')
        allowed = None
        if bundled_path.exists():
            bundled, allowed = read_cases(bundled_path, by_id)
            if 'memory_limit_bytes' not in bundled or 'timeout_seconds' not in bundled:
                raise ValueError('bundled case manifest must specify memory and timeout limits')
            limits.append(bundled)
            case_sources['public_cases_sha256'] = digest(bundled_path)
        cases = allowed
        if args.cases is not None:
            explicit, cases = read_cases(args.cases, by_id)
            limits.append(explicit)
            case_sources['cases_manifest_sha256'] = digest(args.cases)
            if allowed is not None and not {case_key(case) for case in cases} <= {case_key(case) for case in allowed}:
                raise ValueError('explicit cases must be a subset of bundled algorithm/instance/horizon triples')
        if cases is not None:
            permitted_algorithms = {case['algorithm'] for case in cases}
            permitted_instances = {case['instance']['id'] for case in cases}
            if args.algorithms is not None and not set(algorithms) <= permitted_algorithms:
                raise ValueError('algorithm selection contains names absent from the permitted cases')
            if not selected <= permitted_instances:
                raise ValueError('instance selection contains IDs absent from the permitted cases')
            cases = [case for case in cases if case['algorithm'] in algorithms and
                     (case['instance']['id'] in selected if selected else True)]
            if not cases:
                raise ValueError('no permitted algorithm/instance pairs match the selection')
            present_algorithms = {case['algorithm'] for case in cases}
            present_instances = {case['instance']['id'] for case in cases}
            algorithms = [algorithm for algorithm in algorithms if algorithm in present_algorithms]
            instances = [instance for instance in instances if instance['id'] in present_instances]
        else:
            instances = [instance for instance in instances if
                         (instance['id'] in selected if selected else instance['source_kind'] == 'generated')]
            if not instances:
                raise ValueError('instance selection is empty or contains unknown IDs')
            # Unbundled studies retain the original Cartesian campaign protocol.
            cases = [{'instance': instance, 'algorithm': algorithm,
                      'max_depth': max(32, 2 * (instance.get('optimal_length') or 32) + 8)}
                     for instance in instances for algorithm in algorithms]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    memory_cap = min((limit['memory_limit_bytes'] for limit in limits if 'memory_limit_bytes' in limit), default=None)
    timeout_cap = min((limit['timeout_seconds'] for limit in limits if 'timeout_seconds' in limit), default=None)
    if args.memory_gib is None:
        args.memory_gib = min(2, memory_cap / 1024**3) if memory_cap is not None else 2
    if args.timeout_seconds is None:
        args.timeout_seconds = min(120, timeout_cap) if timeout_cap is not None else 120
    for value in (args.duration_days, args.timeout_seconds, args.memory_gib):
        if not math.isfinite(value) or value <= 0:
            parser.error('time and memory limits must be positive and finite')
    if memory_cap is not None and args.memory_gib * 1024**3 > memory_cap:
        parser.error('memory limit exceeds the permitted case manifest limit')
    if timeout_cap is not None and args.timeout_seconds > timeout_cap:
        parser.error('timeout exceeds the permitted case manifest limit')
    verified_inputs = None
    if case_sources:
        try:
            verified_inputs = {str(instance[field]): digest(ROOT / instance[field])
                               for instance in instances for field in ('domain', 'problem')}
            for instance in instances:
                for field in ('domain', 'problem'):
                    if instance.get(field + '_sha256') != verified_inputs[str(instance[field])]:
                        raise ValueError('case input hash does not match the fixture manifest: ' + str(instance[field]))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.dry_run:
        print(json.dumps({'cases_per_round': len(cases), 'algorithms': algorithms,
                          'instances': len(instances), 'duration_days': args.duration_days,
                          'timeout_seconds': args.timeout_seconds, 'memory_gib': args.memory_gib,
                          'rounds': args.rounds, 'explicit_pairs': bool(case_sources)}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'campaign.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another runner owns this output directory')
        inputs = verified_inputs if verified_inputs is not None else {
            str(i[k]): digest(ROOT / i[k]) for i in instances for k in ('domain', 'problem')}
        spec = {'schema_version': 1, 'binary_sha256': digest(args.binary),
                'runner_sha256': digest(__file__), 'inputs_sha256': inputs,
                'cases': cases, 'duration_days': args.duration_days,
                'timeout_seconds': args.timeout_seconds, 'memory_gib': args.memory_gib,
                'rounds': args.rounds, 'seed': args.seed, **case_sources}
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        metadata_path = args.output / 'campaign.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata['spec_sha256'] != spec_hash:
                parser.error('binary, inputs, runner or protocol changed; use a new output directory')
        else:
            metadata = {'started_utc': utc(), 'deadline_epoch': time.time() + args.duration_days * 86400,
                        'spec_sha256': spec_hash, 'spec': spec,
                        'environment': {'platform': platform.platform(), 'python': sys.version,
                                        'cpu_count': os.cpu_count(), 'binary': str(args.binary)},
                        'measurement_note': 'One serial process at a time; repeated rounds share one machine. '
                                            'Timeouts and resource exhaustion are censored; bounded no-plan is not a proof of unsolvability.'}
            metadata['deadline_utc'] = dt.datetime.fromtimestamp(metadata['deadline_epoch'], dt.timezone.utc).isoformat()
            atomic(metadata_path, metadata)
        # Freeze executable and inputs: later developer builds cannot alter a live study.
        frozen = args.output / 'frozen'
        frozen.mkdir(exist_ok=True)
        def snapshot(source, expected):
            target = frozen / expected
            if not target.exists():
                temporary = frozen / (expected + '.tmp')
                shutil.copy2(source, temporary)
                if digest(temporary) != expected:
                    raise ValueError('input changed while freezing the campaign')
                temporary.replace(target)
            if digest(target) != expected:
                raise ValueError('frozen input was modified')
            return target.resolve()
        executable = snapshot(args.binary, spec['binary_sha256'])
        frozen_inputs = {name: snapshot(ROOT / name, expected) for name, expected in inputs.items()}
        raw = args.output / 'raw.jsonl'
        prior = records(raw)
        done = {r['job_id'] for r in prior if r['status'] not in ('interrupted', 'deadline')}
        counts = Counter(r['status'] for r in prior)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        status = {'pid': os.getpid(), 'started_utc': metadata['started_utc'],
                  'deadline_utc': metadata['deadline_utc'], 'status': 'running'}

        def heartbeat(**extra):
            status.update(extra, updated_utc=utc(), completed_jobs=len(done), status_counts=dict(counts))
            atomic(args.output / 'status.json', status)

        round_id = 0
        child = None
        try:
            while not STOP and time.time() < metadata['deadline_epoch'] and (not args.rounds or round_id < args.rounds):
                order = list(range(len(cases)))
                random.Random(args.seed + round_id).shuffle(order)
                for case_id in order:
                    job_id = f'{round_id}:{case_id}'
                    if job_id in done:
                        continue
                    if STOP or time.time() >= metadata['deadline_epoch']:
                        break
                    case = cases[case_id]
                    instance = case['instance']
                    command = [str(executable), '--domain', str(frozen_inputs[instance['domain']]),
                               '--problem', str(frozen_inputs[instance['problem']]), '--algorithm', case['algorithm'],
                               '--heuristic', 'hmax', '--max-depth', str(case['max_depth']), '--repetitions', '1']
                    started = time.monotonic()
                    record = {'job_id': job_id, 'round': round_id, 'instance_id': instance['id'],
                              'family': instance['family'], 'source_kind': instance['source_kind'],
                              'algorithm': case['algorithm'], 'max_depth': case['max_depth'],
                              'reference_cost': instance.get('optimal_length'), 'started_utc': utc(),
                              'timeout_seconds': args.timeout_seconds, 'memory_gib': args.memory_gib, 'spec_sha256': spec_hash,
                              'status': 'error'}
                    logs = args.output / 'jobs'
                    logs.mkdir(exist_ok=True)
                    stem = f'{round_id:05d}-{case_id:04d}'
                    with (logs / (stem + '.stdout')).open('w+') as out, (logs / (stem + '.stderr')).open('w+') as err:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                 stdout=out, stderr=err, start_new_session=True,
                                                 preexec_fn=lambda: child_limits(int(args.memory_gib * 1024**3)),
                                                 env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                        while child.poll() is None:
                            elapsed = time.monotonic() - started
                            heartbeat(active_job=job_id, algorithm=case['algorithm'], instance_id=instance['id'],
                                      round=round_id, child_pid=child.pid, elapsed_seconds=elapsed)
                            if STOP or time.time() >= metadata['deadline_epoch'] or elapsed >= args.timeout_seconds:
                                record['status'] = 'interrupted' if STOP else ('deadline' if time.time() >= metadata['deadline_epoch'] else 'timeout')
                                terminate(child)
                                break
                            try:
                                child.wait(timeout=min(5, max(.01, args.timeout_seconds - elapsed)))
                            except subprocess.TimeoutExpired:
                                pass
                        record['returncode'] = child.returncode
                        record['wall_seconds'] = time.monotonic() - started
                        out.seek(0)
                        lines = [line for line in out if line.strip()]
                        if record['status'] == 'error' and lines:
                            try:
                                measurement = json.loads(lines[-1])
                                if not isinstance(measurement, dict):
                                    raise ValueError('benchmark JSON must be an object')
                                record['measurement'] = measurement
                                reported = measurement.get('status', 'error')
                                valid_exit = (reported, child.returncode) in (
                                    ('solved', 0), ('unsolved', 2), ('error', 1), ('resource_exhausted', 3))
                                record['status'] = reported if valid_exit else 'error'
                                if not valid_exit:
                                    record['error'] = 'benchmark status does not match exit code'
                                if valid_exit and reported == 'error' and measurement.get('phase') == 'replay':
                                    record['status'] = 'invalid_plan'
                                if record['status'] == 'solved' and measurement.get('replay_valid') is not True:
                                    record['status'] = 'invalid_plan'
                            except ValueError:
                                record['error'] = 'benchmark did not emit a valid JSON object'
                        err.seek(0)
                        record['stderr_tail'] = err.read()[-3000:]
                    child = None
                    record['finished_utc'] = utc()
                    with raw.open('a') as stream:
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    counts[record['status']] += 1
                    if record['status'] not in ('interrupted', 'deadline'):
                        done.add(job_id)
                    heartbeat(active_job=None, child_pid=None)
                    if args.plot_every and len(done) and len(done) % args.plot_every == 0:
                        refresh_figures(args.output)
                round_id += 1
            heartbeat(status='interrupted' if STOP else ('deadline_reached' if time.time() >= metadata['deadline_epoch'] else 'complete'),
                      active_job=None, child_pid=None)
        finally:
            if child is not None:
                terminate(child)
        if raw.exists():
            refresh_figures(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
