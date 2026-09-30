#!/usr/bin/env python3
"""Process-level regression checks for durable result handling."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.binary = self.folder / 'planner'
        self.output = self.folder / 'results'

    def program(self, code):
        self.binary.write_text('#!/usr/bin/env python3\n' + code + '\n')
        self.binary.chmod(0o755)

    def run_campaign(self, timeout='2'):
        return subprocess.run([sys.executable, str(ROOT / 'benchmarks/serial_campaign.py'),
                               '--binary', str(self.binary), '--output', str(self.output),
                               '--instances', 'blocks-sussman', '--algorithms', 'astar',
                               '--rounds', '1', '--timeout-seconds', timeout, '--plot-every', '0'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)

    def rows(self):
        return [json.loads(line) for line in (self.output / 'raw.jsonl').read_text().splitlines()]

    def test_no_plan_exit_is_not_a_crash_and_resume_skips_it(self):
        self.program('import json\nprint(json.dumps({"status":"unsolved"}))\nraise SystemExit(2)')
        first = self.run_campaign()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self.rows()[0]['status'], 'unsolved')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(len(self.rows()), 1)

    def test_invalid_plan_cannot_be_reported_as_solved(self):
        self.program('print(\'{"status":"solved","replay_valid":false}\')')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'invalid_plan')

    def test_real_benchmark_replay_error_schema_is_invalid_plan(self):
        self.program('print(\'{"status":"error","phase":"replay","replay_valid":false,"error":"independent IR replay: goal fails"}\')\nraise SystemExit(1)')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'invalid_plan')

    def test_allocation_failure_is_resource_exhausted_and_resume_skips_it(self):
        self.program('print(\'{"status":"resource_exhausted","phase":"planning","error":"std::bad_alloc"}\')\nraise SystemExit(3)')
        first = self.run_campaign()
        self.assertEqual(first.returncode, 0, first.stderr)
        row = self.rows()[0]
        self.assertEqual(row['status'], 'resource_exhausted')
        self.assertEqual(row['measurement']['error'], 'std::bad_alloc')
        self.assertEqual(row['memory_gib'], 2)
        state = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(state['status_counts'], {'resource_exhausted': 1})
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(len(self.rows()), 1)

    def test_replay_allocation_failure_is_not_an_invalid_plan(self):
        self.program('print(\'{"status":"resource_exhausted","phase":"replay","replay_valid":false,"error":"std::bad_alloc"}\')\nraise SystemExit(3)')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'resource_exhausted')

    def test_resource_status_requires_matching_exit(self):
        self.program('print(\'{"status":"resource_exhausted","error":"std::bad_alloc"}\')\nraise SystemExit(1)')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'error')
        self.assertEqual(self.rows()[0]['error'], 'benchmark status does not match exit code')

    def test_legacy_allocation_error_is_not_silently_reclassified(self):
        self.program('print(\'{"status":"error","phase":"planning","error":"std::bad_alloc"}\')\nraise SystemExit(1)')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'error')

    def test_non_object_json_is_recorded_as_error(self):
        self.program('print("[]")')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'error')
        self.assertEqual(self.rows()[0]['error'], 'benchmark did not emit a valid JSON object')

    def test_timeout_and_frozen_binary(self):
        self.program('import time\ntime.sleep(20)')
        self.assertEqual(self.run_campaign('.1').returncode, 0)
        self.assertEqual(self.rows()[0]['status'], 'timeout')
        self.assertLess(self.rows()[0]['wall_seconds'], 4)
        metadata = json.loads((self.output / 'campaign.json').read_text())
        binary = self.output / 'frozen' / metadata['spec']['binary_sha256']
        self.assertEqual(binary.read_bytes(), self.binary.read_bytes())

    def test_changed_binary_requires_separate_result_directory(self):
        self.program('print(\'{"status":"solved","replay_valid":true}\')')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.program('print(\'{"status":"unsolved"}\')')
        self.assertNotEqual(self.run_campaign().returncode, 0)
        self.assertEqual(len(self.rows()), 1)

    def test_incomplete_final_write_repaired_without_duplicate(self):
        self.program('print(\'{"status":"solved","replay_valid":true}\')')
        self.assertEqual(self.run_campaign().returncode, 0)
        with (self.output / 'raw.jsonl').open('a') as stream:
            stream.write('{"job_id":')
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertEqual(len(self.rows()), 1)

    def test_complete_final_record_missing_newline_is_repaired(self):
        self.program('print(\'{"status":"solved","replay_valid":true}\')')
        self.assertEqual(self.run_campaign().returncode, 0)
        raw = self.output / 'raw.jsonl'
        raw.write_bytes(raw.read_bytes().rstrip(b'\n'))
        self.assertEqual(self.run_campaign().returncode, 0)
        self.assertTrue(raw.read_bytes().endswith(b'\n'))
        self.assertEqual(len(self.rows()), 1)


class CuratedCaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.benchmarks = self.folder / 'benchmarks'
        self.benchmarks.mkdir()
        self.runner = self.benchmarks / 'serial_campaign.py'
        shutil.copy2(ROOT / 'benchmarks/serial_campaign.py', self.runner)
        manifest = json.loads((ROOT / 'benchmarks/manifest.json').read_text())
        self.instances = manifest['instances'][:2]
        (self.benchmarks / 'manifest.json').write_text(json.dumps({'instances': self.instances}))
        for instance in self.instances:
            for field in ('domain', 'problem'):
                target = self.folder / instance[field]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / instance[field], target)
        self.document = {'schema_version': 1, 'memory_limit_bytes': 2 * 1024**3, 'timeout_seconds': 120,
                         'cases': [{'algorithm': 'bfs', 'instance': self.instances[0], 'max_depth': 3},
                                   {'algorithm': 'astar', 'instance': self.instances[1], 'max_depth': 7}]}
        self.public_cases = self.benchmarks / 'public_cases.json'
        self.write_cases()

    def write_cases(self):
        self.public_cases.write_text(json.dumps(self.document))

    def run_selection(self, *arguments):
        return subprocess.run([sys.executable, str(self.runner), '--dry-run', *arguments],
                              capture_output=True, text=True, timeout=10)

    def test_default_bundle_keeps_explicit_pairs_and_qualified_horizons(self):
        preview = self.run_selection()
        self.assertEqual(preview.returncode, 0, preview.stderr)
        data = json.loads(preview.stdout)
        self.assertEqual(data['cases_per_round'], 2)
        self.assertEqual(set(data['algorithms']), {'bfs', 'astar'})
        self.assertTrue(data['explicit_pairs'])
        binary = self.folder / 'planner'
        binary.write_text('#!/usr/bin/env python3\nprint(\'{"status":"solved","replay_valid":true}\')\n')
        binary.chmod(0o755)
        output = self.folder / 'results'
        execution = subprocess.run([sys.executable, str(self.runner), '--binary', str(binary),
                                    '--output', str(output), '--rounds', '1', '--plot-every', '0'],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
        self.assertEqual({(row['algorithm'], row['instance_id'], row['max_depth']) for row in rows},
                         {('bfs', self.instances[0]['id'], 3), ('astar', self.instances[1]['id'], 7)})
        captured = json.loads((output / 'campaign.json').read_text())
        self.assertIn('public_cases_sha256', captured['spec'])

    def test_algorithm_and_instance_filters_only_narrow_pairs(self):
        selected = self.run_selection('--algorithms', 'bfs', '--instances',
                                      ','.join(instance['id'] for instance in self.instances))
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(json.loads(selected.stdout)['cases_per_round'], 1)
        excluded = self.run_selection('--algorithms', 'bfs', '--instances', self.instances[1]['id'])
        self.assertNotEqual(excluded.returncode, 0)
        self.assertIn('no permitted', excluded.stderr)
        self.assertNotEqual(self.run_selection('--algorithms', 'satplan').returncode, 0)

    def test_alternate_cases_cannot_add_pairs_or_change_horizons(self):
        explicit = self.folder / 'selected.json'
        subset = copy.deepcopy(self.document)
        subset['cases'] = subset['cases'][:1]
        explicit.write_text(json.dumps(subset))
        selected = self.run_selection('--cases', str(explicit))
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(json.loads(selected.stdout)['cases_per_round'], 1)
        for field, value in [('algorithm', 'astar'), ('max_depth', 4)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(subset)
                changed['cases'][0][field] = value
                explicit.write_text(json.dumps(changed))
                rejected = self.run_selection('--cases', str(explicit))
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn('subset', rejected.stderr)

    def test_bundle_rejects_inconsistent_metadata_and_invalid_cases(self):
        changes = [lambda value: value['cases'].append(copy.deepcopy(value['cases'][0])),
                   lambda value: value['cases'][0]['instance'].update(domain='wrong.pddl'),
                   lambda value: value['cases'][0].update(instance_id=self.instances[1]['id']),
                   lambda value: value['cases'][0].update(algorithm='unknown'),
                   lambda value: value['cases'][0].update(max_depth=True),
                   lambda value: value['cases'][0].update(max_depth=-1),
                   lambda value: value['cases'][0].update(max_depth=2**64)]
        original = copy.deepcopy(self.document)
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.document = copy.deepcopy(original)
                change(self.document)
                self.write_cases()
                rejected = self.run_selection()
                self.assertNotEqual(rejected.returncode, 0)
                self.assertNotIn('Traceback', rejected.stderr)

    def test_bundle_rejects_modified_fixture_bytes(self):
        problem = self.folder / self.instances[0]['problem']
        problem.write_text(problem.read_text() + '\n; modified after qualification\n')
        rejected = self.run_selection()
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('input hash does not match', rejected.stderr)

    def test_bundle_caps_memory_and_timeout_but_allows_lower_values(self):
        for arguments in [('--memory-gib', '2.01'), ('--timeout-seconds', '121')]:
            with self.subTest(arguments=arguments):
                rejected = self.run_selection(*arguments)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn('exceeds', rejected.stderr)
        selected = self.run_selection('--memory-gib', '.5', '--timeout-seconds', '5')
        self.assertEqual(selected.returncode, 0, selected.stderr)
        data = json.loads(selected.stdout)
        self.assertEqual(data['memory_gib'], .5)
        self.assertEqual(data['timeout_seconds'], 5)

    def test_unbundled_defaults_remain_cartesian_and_explicit_cases_are_optional(self):
        self.public_cases.unlink()
        full = self.run_selection('--algorithms', 'bfs,astar', '--memory-gib', '3', '--timeout-seconds', '121')
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(json.loads(full.stdout)['cases_per_round'], 4)
        explicit = self.folder / 'selected.json'
        explicit.write_text(json.dumps(self.document))
        limited = self.run_selection('--cases', str(explicit))
        self.assertEqual(limited.returncode, 0, limited.stderr)
        self.assertEqual(json.loads(limited.stdout)['cases_per_round'], 2)


if __name__ == '__main__':
    unittest.main()
