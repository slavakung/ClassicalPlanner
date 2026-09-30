import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from export_public import export
from public_results import ALGORITHMS, build_matrix, read_snapshot, terminal


class PublicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'benchmarks/manifest.json').read_text())
        cls.hashes = {i[k]: hashlib.sha256((ROOT / i[k]).read_bytes()).hexdigest()
                      for i in cls.manifest['instances'] for k in ('domain', 'problem')}
        cls.cases = [{'algorithm': a, 'instance': i, 'max_depth': 32}
                     for i in cls.manifest['instances'] for a in ALGORITHMS]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source, self.output = self.folder / 'campaign', self.folder / 'public'
        self.source.mkdir()
        now = dt.datetime.now(dt.timezone.utc)
        self.campaign = {'started_utc': (now - dt.timedelta(days=8)).isoformat(),
                         'deadline_epoch': (now - dt.timedelta(days=1)).timestamp(),
                         'deadline_utc': (now - dt.timedelta(days=1)).isoformat(), 'spec_sha256': 'captured-spec',
                         'spec': {'cases': self.cases, 'inputs_sha256': self.hashes, 'rounds': 0,
                                  'timeout_seconds': 120, 'memory_gib': 2, 'duration_days': 7}}
        self.state = {'status': 'running', 'active_job': '0:1'}
        self.context = {'exclude_timing_intervals': [{'start_utc': '2026-09-16T00:00:00Z',
                                                     'end_utc': '2026-09-17T00:00:00Z', 'reason': 'research overlap'}]}

    def row(self, **changes):
        value = {'job_id': '0:0', 'round': 0, 'algorithm': 'bfs', 'instance_id': self.manifest['instances'][0]['id'],
                 'status': 'solved', 'spec_sha256': 'captured-spec', 'wall_seconds': .3,
                 'started_utc': '2026-09-16T01:00:00Z', 'finished_utc': '2026-09-16T01:00:01Z',
                 'measurement': {'algorithm': 'bfs', 'status': 'solved', 'replay_valid': True, 'unit_action_costs': True,
                                 'plan_cost': 6, 'plan_length': 6, 'plan_actions': ['action'] * 6,
                                 'planning_seconds': .2, 'parse_seconds': .02, 'grounding_seconds': .03,
                                 'replay_seconds': .01, 'domain_file': str(ROOT / 'private/domain'),
                                 'problem_file': str(ROOT / 'private/problem')}}
        value.update(changes)
        return value

    def capture(self, rows):
        for name, value in [('campaign.json', self.campaign), ('status.json', self.state),
                            ('measurement_context.json', self.context)]:
            (self.source / name).write_text(json.dumps(value))
        (self.source / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return read_snapshot(self.source)

    def matrix(self, rows, hashes=None):
        return build_matrix(self.manifest, self.capture(rows), self.hashes if hashes is None else hashes)[0]

    def test_full_26_by_48_matrix_includes_pending_and_unattempted(self):
        matrix = self.matrix([self.row()])
        self.assertEqual(len(matrix['cases']), 1248)
        self.assertEqual(matrix['summary']['publication_state'], 'preliminary')
        self.assertEqual(matrix['summary']['pending_cases'], 1)
        self.assertEqual(sum(r['outcome'] == 'not_attempted' for r in matrix['cases']), 1246)
        self.assertEqual(matrix['cases'][0]['optimality'], 'proven_reference_match_all_solutions')

    def test_excluded_timing_preserves_original_replay_and_optimality(self):
        snapshot = self.capture([self.row()])
        context = (self.source / 'measurement_context.json').read_bytes()
        matrix, annotated = build_matrix(self.manifest, snapshot, self.hashes)
        row = matrix['cases'][0]
        self.assertEqual(row['feasibility'], 'feasible')
        self.assertEqual(row['reference_matching_solutions'], 1)
        self.assertEqual(row['eligible_timing_samples'], 0)
        self.assertIsNone(row['timing']['planning_seconds']['median'])
        self.assertFalse(annotated[0]['timing_eligible'])
        self.assertNotIn('timing_eligible', snapshot['rows'][0])
        self.assertEqual((self.source / 'measurement_context.json').read_bytes(), context)

    def test_hash_replay_unit_cost_and_plan_length_gate_proofs(self):
        changed = dict(self.hashes)
        changed[self.manifest['instances'][0]['problem']] = 'wrong'
        self.assertEqual(self.matrix([self.row()], changed)['cases'][0]['optimality'], 'not_proven')
        for field, value in [('replay_valid', False), ('plan_actions', ['truncated']), ('algorithm', 'different')]:
            with self.subTest(field=field):
                row = self.row()
                row['measurement'][field] = value
                result = self.matrix([row])['cases'][0]
                self.assertEqual(result['feasibility'], 'not_proven')
                self.assertEqual(result['outcome'], 'invalid_plan')
        row = self.row()
        row['measurement']['unit_action_costs'] = False
        self.assertEqual(self.matrix([row])['cases'][0]['optimality'], 'not_proven')

    def test_changed_reference_cannot_relabel_suboptimal_solution_as_optimal(self):
        changed = copy.deepcopy(self.manifest)
        changed['instances'][0]['optimal_length'] = 8
        row = self.row()
        row['measurement'].update(plan_cost=8, plan_length=8, plan_actions=['action'] * 8)
        matrix, _ = build_matrix(changed, self.capture([row]), self.hashes)
        self.assertEqual(matrix['cases'][0]['feasibility'], 'feasible')
        self.assertEqual(matrix['cases'][0]['optimality'], 'not_proven')

    def test_failed_attempts_never_prove_infeasibility(self):
        for status in ('unsolved', 'timeout', 'error', 'resource_exhausted'):
            with self.subTest(status=status):
                row = self.matrix([self.row(status=status, measurement={})])['cases'][0]
                self.assertEqual(row['infeasibility'], 'not_proven')
                self.assertEqual(row['feasibility'], 'not_proven')
                self.assertEqual(row['optimality'], 'not_proven')
                self.assertIsNone(row['timing']['planning_seconds']['median'])

    def test_resource_exhaustion_is_censored_and_preserved_without_rewriting_history(self):
        resource = self.row(status='resource_exhausted', measurement={'status': 'resource_exhausted',
                            'phase': 'planning', 'error': 'std::bad_alloc'},
                            started_utc='2026-09-18T01:00:00Z', finished_utc='2026-09-18T01:00:01Z')
        legacy = self.row(job_id='1:0', round=1, status='error',
                          measurement={'status': 'error', 'error': 'std::bad_alloc'})
        snapshot = self.capture([resource, legacy])
        original = (self.source / 'raw.jsonl').read_bytes()
        matrix, _ = build_matrix(self.manifest, snapshot, self.hashes)
        row = matrix['cases'][0]
        self.assertEqual(row['status_counts'], {'resource_exhausted': 1, 'error': 1})
        self.assertTrue(row['timing_by_outcome']['resource_exhausted']['censored'])
        self.assertEqual(row['timing_by_outcome']['resource_exhausted']['eligible_process_wall_seconds']['n'], 1)
        self.assertEqual(row['eligible_timing_samples'], 0)
        self.assertEqual(row['outcome'], 'mixed')
        self.assertEqual((self.source / 'raw.jsonl').read_bytes(), original)
        self.assertEqual(self.matrix([resource])['cases'][0]['outcome'], 'resource_exhausted')

    def test_repeats_preserve_mixed_quality_without_double_counting_resume(self):
        second = self.row(job_id='1:0', round=1, started_utc='2026-09-18T01:00:00Z', finished_utc='2026-09-18T01:00:01Z')
        second['measurement'].update(plan_cost=8, plan_length=8, plan_actions=['action'] * 8)
        matrix = self.matrix([self.row(status='interrupted', measurement={}), self.row(), second])
        row = matrix['cases'][0]
        self.assertEqual(row['recorded_repetitions'], 2)
        self.assertEqual(row['optimality'], 'proven_reference_match_some_solutions')
        self.assertEqual(row['eligible_timing_samples'], 1)
        self.assertEqual(matrix['summary']['superseded_attempts'], 1)

    def test_passed_deadline_is_not_completion_and_final_guard_is_safe(self):
        self.capture([self.row()])
        self.assertFalse(terminal(self.campaign, self.state))
        with self.assertRaises(ValueError):
            export(self.source, self.output, figures=False, require_complete=True)
        self.assertFalse(self.output.exists())
        self.state['status'] = 'interrupted'
        self.assertFalse(terminal(self.campaign, self.state))
        self.state['status'] = 'deadline_reached'
        self.assertTrue(terminal(self.campaign, self.state))

    def test_torn_tail_is_visible_and_blocks_final_export(self):
        self.state['status'] = 'deadline_reached'
        self.capture([self.row()])
        with (self.source / 'raw.jsonl').open('a') as stream:
            stream.write('{"job_id":')
        snapshot = read_snapshot(self.source)
        self.assertTrue(snapshot['snapshot']['incomplete_trailing_record_omitted'])
        self.assertFalse(build_matrix(self.manifest, snapshot, self.hashes)[0]['summary']['campaign_finished'])
        with self.assertRaises(ValueError):
            export(self.source, self.output, figures=False, require_complete=True)

    def test_public_source_bundle_is_portable_and_all_cells_exported(self):
        self.capture([self.row()])
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        result = export(self.source, self.output, figures=False)
        self.assertEqual(result['matrix_cases'], 1248)
        self.assertEqual(result['publication_state'], 'preliminary')
        self.assertFalse(result['campaign_window_complete'])
        self.assertEqual(len((self.output / 'results/matrix.csv').read_text().splitlines()), 1249)
        for name in ('LICENSE', 'CMakeLists.txt', 'benchmarks/serial_campaign.py'):
            self.assertTrue((self.output / name).exists())
        self.assertFalse((self.output / 'build').exists())
        self.assertFalse((self.output / 'benchmarks/results').exists())
        self.assertNotIn('/home/', (self.output / 'results/raw.jsonl').read_text())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.state['status'] = 'deadline_reached'
        self.capture([self.row()])
        final = export(self.source, self.output, figures=False, require_complete=True)
        self.assertEqual(final['publication_state'], 'final')
        self.assertTrue(final['campaign_finished'])
        self.assertTrue(final['campaign_window_complete'])

    def test_refuses_overlapping_or_unowned_destination(self):
        self.capture([])
        for target in (self.source, self.source / 'publication', ROOT):
            with self.subTest(target=target), self.assertRaises(ValueError):
                export(self.source, target, figures=False)
        self.output.mkdir()
        (self.output / 'user-file').write_text('keep')
        with self.assertRaises(ValueError):
            export(self.source, self.output, figures=False)
        self.assertEqual((self.output / 'user-file').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
