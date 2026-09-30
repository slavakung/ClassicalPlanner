#!/usr/bin/env python3
"""Process tests for strict qualification, hard limits, and safe resumption."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BoundedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.binary = self.folder / 'planner'
        self.output = self.folder / 'results'
        self.manifest = self.folder / 'cases.json'
        instance = json.loads((ROOT / 'benchmarks/manifest.json').read_text())['instances'][0]
        self.cases = {'schema_version': 1, 'cases': [
            {'instance': instance, 'algorithm': 'astar', 'max_depth': 32}],
            'memory_limit_bytes': 64 * 1024**2, 'timeout_seconds': 2}
        self.write_cases()

    def write_cases(self):
        self.manifest.write_text(json.dumps(self.cases))

    def program(self, code):
        self.binary.write_text('#!/usr/bin/env python3\n' + code + '\n')
        self.binary.chmod(0o755)

    def run_bounded(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'benchmarks/run_bounded.py'),
            '--binary', str(self.binary), '--cases', str(self.manifest), '--output', str(self.output),
            *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)

    def rows(self):
        return [json.loads(line) for line in (self.output / 'raw.jsonl').read_text().splitlines()]

    def assert_runs(self, *extra):
        result = self.run_bounded(*extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.rows()[-1]

    def test_solved_is_replayed_capped_and_resumable(self):
        self.program('import json, os, resource\nprint(json.dumps({"status":"solved","replay_valid":True,"as_limit":resource.getrlimit(resource.RLIMIT_AS),"core_limit":resource.getrlimit(resource.RLIMIT_CORE),"nice":os.nice(0)}))')
        row = self.assert_runs()
        self.assertTrue(row['qualified'])
        self.assertEqual(row['measurement']['as_limit'], [64 * 1024**2] * 2)
        self.assertEqual(row['measurement']['core_limit'], [0, 0])
        self.assertGreaterEqual(row['measurement']['nice'], 10)
        self.assertEqual(row['memory_limit_kind'], 'RLIMIT_AS')
        self.assert_runs()
        self.assertEqual(len(self.rows()), 1)

    def test_bounded_unsolved_is_qualified_and_exit_is_valid(self):
        self.program('print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        self.assertTrue(self.assert_runs()['qualified'])
        self.assertEqual(self.rows()[0]['status'], 'unsolved')

    def test_invalid_solution_is_rejected(self):
        self.program('print(\'{"status":"solved","replay_valid":false}\')')
        row = self.assert_runs()
        self.assertEqual(row['status'], 'invalid_plan')
        self.assertFalse(row['qualified'])

    def test_resource_exhaustion_is_distinct(self):
        self.program('import json\ntry:\n data=bytearray(256*1024**2)\nexcept MemoryError:\n print(json.dumps({"status":"resource_exhausted"}))\n raise SystemExit(3)\nraise SystemExit(42)')
        row = self.assert_runs()
        self.assertEqual(row['status'], 'resource_exhausted')
        self.assertFalse(row['qualified'])

    def test_nonresource_error_is_not_hidden(self):
        self.program('print(\'{"status":"error","error":"parser defect"}\')\nraise SystemExit(1)')
        row = self.assert_runs()
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['measurement']['error'], 'parser defect')
        self.assertFalse(row['qualified'])

    def test_wrong_exit_is_rejected(self):
        self.program('print(\'{"status":"solved","replay_valid":true}\')\nraise SystemExit(1)')
        row = self.assert_runs()
        self.assertEqual(row['status'], 'error')
        self.assertFalse(row['qualified'])

    def test_timeout_reports_sampled_memory_and_is_not_qualified(self):
        self.program('import time\ndata=bytearray(8*1024**2)\ntime.sleep(30)')
        row = self.assert_runs('--timeout-seconds', '.2')
        self.assertEqual(row['status'], 'timeout')
        self.assertFalse(row['qualified'])
        self.assertLess(row['wall_seconds'], 2)
        if sys.platform.startswith('linux'):
            self.assertGreater(row['peak_rss_kib'], 8192)
            self.assertGreater(row['peak_virtual_kib'], row['peak_rss_kib'])

    def test_explicit_cases_never_expand_algorithms(self):
        self.cases['cases'].append({**self.cases['cases'][0], 'algorithm': 'bfs'})
        self.write_cases()
        self.program('print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        self.assert_runs()
        self.assertEqual([r['algorithm'] for r in self.rows()], ['astar', 'bfs'])

    def test_changed_binary_or_protocol_cannot_reuse_evidence(self):
        self.program('print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        self.assert_runs()
        self.assertNotEqual(self.run_bounded('--timeout-seconds', '3').returncode, 0)
        self.program('print(\'{"status":"solved","replay_valid":true}\')')
        self.assertNotEqual(self.run_bounded().returncode, 0)
        self.assertEqual(len(self.rows()), 1)

    def test_invalid_algorithm_and_oversized_depth_fail_before_execution(self):
        self.program('raise SystemExit(42)')
        for field, value in (('algorithm', 'typo'), ('max_depth', 2**64)):
            original = self.cases['cases'][0][field]
            self.cases['cases'][0][field] = value
            self.write_cases()
            self.assertNotEqual(self.run_bounded().returncode, 0)
            self.assertFalse((self.output / 'raw.jsonl').exists())
            self.cases['cases'][0][field] = original

    def test_interrupted_case_is_retried_on_resume(self):
        marker = self.folder / 'first-run'
        self.program('from pathlib import Path\nimport time\n' +
                     f'marker=Path({str(marker)!r})\n' +
                     'if not marker.exists():\n marker.touch()\n time.sleep(30)\n' +
                     'print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        child = subprocess.Popen([sys.executable, str(ROOT / 'benchmarks/run_bounded.py'),
            '--binary', str(self.binary), '--cases', str(self.manifest), '--output', str(self.output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.kill_if_present, child.pid)
        deadline = time.monotonic() + 5
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(marker.exists())
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=5)
        self.assertEqual(child.returncode, 130)
        self.assertEqual(self.rows()[0]['status'], 'interrupted')
        self.assertFalse(self.rows()[0]['qualified'])
        self.assertEqual(self.assert_runs()['status'], 'unsolved')
        self.assertEqual(len(self.rows()), 2)
        state = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(state['completed_cases'], 1)
        self.assertEqual(state['status_counts'], {'unsolved': 1})

    def test_partial_last_write_is_repaired(self):
        self.program('print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        self.assert_runs()
        with (self.output / 'raw.jsonl').open('a') as stream:
            stream.write('{"case_id":')
        self.assert_runs()
        self.assertEqual(len(self.rows()), 1)

    def test_non_object_or_nonfinite_json_is_rejected(self):
        self.program('print(\'{"status":"solved","replay_valid":true,"peak_rss_kib":NaN}\')')
        row = self.assert_runs()
        self.assertFalse(row['qualified'])
        self.assertEqual(row['status'], 'error')

    def test_peak_reaching_cap_is_rejected(self):
        self.program('print(\'{"status":"solved","replay_valid":true,"peak_rss_kib":65536}\')')
        row = self.assert_runs()
        self.assertEqual(row['status'], 'resource_exhausted')
        self.assertFalse(row['qualified'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires /proc')
    def test_orphaned_descendants_are_terminated(self):
        pidfile = self.folder / 'descendant.pid'
        self.program('import os,time\npid=os.fork()\nif pid==0:\n time.sleep(30)\n os._exit(0)\n' +
                     f'open({str(pidfile)!r},"w").write(str(pid))\n' +
                     'print(\'{"status":"unsolved"}\')\nraise SystemExit(2)')
        self.assert_runs()
        pid = int(pidfile.read_text())
        self.addCleanup(self.kill_if_present, pid)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            status = Path(f'/proc/{pid}/status')
            if not status.exists() or 'State:\tZ' in status.read_text():
                break
            time.sleep(.01)
        else:
            self.fail('descendant survived the runner cleanup')

    @staticmethod
    def kill_if_present(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class PublicAllowlistTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        spec = importlib.util.spec_from_file_location('bounded_runner', ROOT / 'benchmarks/run_bounded.py')
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.runner.ROOT = self.folder
        (self.folder / 'benchmarks').mkdir()
        (self.folder / 'domain.pddl').write_text('domain')
        (self.folder / 'problem.pddl').write_text('problem')
        self.case = {'instance': {'id': 'case', 'domain': 'domain.pddl', 'problem': 'problem.pddl',
                                'domain_sha256': self.runner.digest(self.folder / 'domain.pddl'),
                                'problem_sha256': self.runner.digest(self.folder / 'problem.pddl')},
                     'algorithm': 'bfs', 'max_depth': 32}
        (self.folder / 'benchmarks/public_cases.json').write_text(json.dumps({
            'cases': [self.case], 'memory_limit_bytes': 2 * 1024**3, 'timeout_seconds': 120}))

    def test_narrower_caps_and_approved_case_are_permitted(self):
        self.runner.enforce_public_allowlist([self.case], 60, 512 * 1024**2)

    def test_pair_and_depth_expansion_are_rejected(self):
        for field, value in (('algorithm', 'astar'), ('max_depth', 64)):
            with self.assertRaises(ValueError):
                self.runner.enforce_public_allowlist([{**self.case, field: value}], 120, 2 * 1024**3)

    def test_cap_expansion_is_rejected(self):
        for timeout, memory in ((121, 2 * 1024**3), (120, 3 * 1024**3)):
            with self.assertRaises(ValueError):
                self.runner.enforce_public_allowlist([self.case], timeout, memory)

    def test_input_modification_is_rejected(self):
        (self.folder / 'problem.pddl').write_text('different task')
        with self.assertRaises(ValueError):
            self.runner.enforce_public_allowlist([self.case], 120, 2 * 1024**3)


if __name__ == '__main__':
    unittest.main()
