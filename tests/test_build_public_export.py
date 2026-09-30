import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from export_build_public import (ALGORITHMS, MANIFEST, MEMORY_LIMIT_BYTES, export,
                                 fixture_hashes, qualification_source_hashes, source_files)


class BuildPublicExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.repository = self.folder / 'source'
        self.repository.mkdir()
        for source in source_files(ROOT):
            target = self.repository / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        # A stub keeps these exporter tests independent of runner implementation.
        (self.repository / 'benchmarks/run_bounded.py').write_text('# test runner fixture\n')
        self.output = self.folder / 'build-public'
        self.evidence = self.folder / 'qualification.json'
        manifest = json.loads((self.repository / 'benchmarks/manifest.json').read_text())
        self.instances = manifest['instances']
        self.qualification = {
            'schema_version': 1, 'memory_limit_bytes': MEMORY_LIMIT_BYTES, 'timeout_seconds': 120,
            'source_sha256': qualification_source_hashes(self.repository),
            'inputs_sha256': fixture_hashes(self.repository, manifest),
            'cases': [{'algorithm': algorithm, 'instance_id': self.instances[0]['id'],
                       'status': 'solved', 'qualified': True, 'max_depth': 32,
                       'peak_rss_kib': 4096, 'peak_virtual_kib': 16384,
                       'measurement': {'replay_valid': True}}
                      for algorithm in ALGORITHMS],
        }

    def write_evidence(self):
        self.evidence.write_text(json.dumps(self.qualification))

    def publish(self):
        self.write_evidence()
        return export(self.evidence, self.output, repository=self.repository)

    def test_only_sources_tests_and_explicitly_qualified_benchmarks_are_exported(self):
        for name in ('SOURCE_SNAPSHOT.json', 'build/benchmark_planner', 'public/results/raw.jsonl',
                     'benchmarks/results/status.json', 'benchmarks/frozen/private.json',
                     'docs/CAMPAIGN.md', 'scripts/run_serial_week.sh', 'tests/__pycache__/cached.pyc'):
            target = self.repository / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('must stay private')
        for index, status in enumerate(('timeout', 'resource_exhausted', 'error'), start=1):
            self.qualification['cases'].append({'algorithm': ALGORITHMS[0],
                'instance_id': self.instances[index]['id'], 'qualified': False, 'status': status})
        self.qualification.update(fresh_checked_cases=26, historical_excluded_cases=3)
        record = self.publish()
        cases = json.loads((self.output / 'benchmarks/public_cases.json').read_text())
        self.assertEqual(len(cases['cases']), 26)
        self.assertEqual(record['evaluated_case_count'], 29)
        self.assertEqual({row['algorithm'] for row in cases['cases']}, set(ALGORITHMS))
        self.assertTrue(all(row['instance_id'] == row['instance']['id'] for row in cases['cases']))
        proof = cases['cases'][0]['qualification']
        self.assertEqual(proof['peak_rss_kib'], 4096)
        self.assertEqual(proof['memory_limit_bytes'], MEMORY_LIMIT_BYTES)
        self.assertTrue(proof['replay_valid'])
        for name in ('CMakeLists.txt', 'LICENSE', 'NOTICE', 'Planner.h', 'SearchStorage.h',
                     'tests.cpp', 'tests/test_public_export.py', 'tests/search_regressions.cpp',
                     'benchmarks/run_bounded.py', 'benchmarks/serial_campaign.py'):
            self.assertIn(name, record['files'])
        for name in record['files']:
            path = self.output / name
            self.assertEqual(record['files'][name]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotEqual('must stay private', path.read_text())
        self.assertNotIn(MANIFEST, record['files'])
        self.assertFalse((self.output / 'qualification.json').exists())
        self.assertEqual(len(json.loads((self.output / 'benchmarks/manifest.json').read_text())['instances']), 48)
        self.assertIn('--cases benchmarks/public_cases.json', (self.output / 'README.md').read_text())
        self.assertNotIn('run_serial_week.sh', (self.output / 'README.md').read_text())
        self.assertIn('26 pairs; 3 pairs', (self.output / 'README.md').read_text())

    def test_unqualified_failures_cannot_be_selected_by_status_or_flag(self):
        original = copy.deepcopy(self.qualification)
        for update in ({'status': 'error'}, {'status': 'resource_exhausted'}, {'status': 'timeout'},
                       {'peak_rss_kib': MEMORY_LIMIT_BYTES // 1024}, {'peak_rss_kib': None},
                       {'peak_virtual_kib': float('nan')}, {'measurement': {'replay_valid': False}},
                       {'memory_limit_bytes': MEMORY_LIMIT_BYTES + 1}, {'memory_limit_bytes': 0},
                       {'memory_limit_bytes': 8 * 1024 ** 2}, {'timeout_seconds': 121}):
            with self.subTest(update=update):
                self.qualification = copy.deepcopy(original)
                self.qualification['cases'][0].update(update)
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(self.output.exists())
        self.qualification = copy.deepcopy(original)
        self.qualification['cases'][0].update(status='unsolved', measurement={})
        self.qualification['cases'][1]['memory_limit_bytes'] = 512 * 1024 ** 2
        self.qualification['cases'][1]['timeout_seconds'] = 30
        self.assertEqual(self.publish()['qualified_case_count'], 26)

    def test_stale_build_or_fixture_evidence_is_rejected(self):
        for filename in ('Planner.h', self.instances[0]['problem']):
            with self.subTest(filename=filename):
                target = self.repository / filename
                original = target.read_bytes()
                target.write_bytes(original + b'\n')
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(self.output.exists())
                target.write_bytes(original)
        self.qualification['source_sha256'].pop('Planner.h')
        with self.assertRaises(ValueError):
            self.publish()

    def test_unknown_duplicate_and_escaping_cases_are_rejected(self):
        original = copy.deepcopy(self.qualification)
        self.qualification['cases'].append(copy.deepcopy(self.qualification['cases'][0]))
        with self.assertRaises(ValueError):
            self.publish()
        self.qualification = copy.deepcopy(original)
        self.qualification['cases'][0]['instance_id'] = '../escape'
        with self.assertRaises(ValueError):
            self.publish()
        self.qualification = copy.deepcopy(original)
        manifest_path = self.repository / 'benchmarks/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['instances'][0]['domain'] = '../../outside.pddl'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.publish()

    def test_path_overlap_unowned_destination_and_symlinks_are_rejected(self):
        self.write_evidence()
        for destination in (self.repository, self.repository / 'build-public', self.folder,
                            self.evidence, self.evidence / 'output'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                export(self.evidence, destination, repository=self.repository)
        self.output.mkdir()
        keep = self.output / 'user-file'
        keep.write_text('keep this')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(keep.read_text(), 'keep this')
        shutil.rmtree(self.output)
        self.output.symlink_to(self.repository, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.publish()
        self.output.unlink()
        (self.repository / 'tests/escape.py').symlink_to(self.evidence)
        with self.assertRaises(ValueError):
            self.publish()

    def test_owned_export_can_be_refreshed_but_local_edits_are_preserved(self):
        first = self.publish()
        second = self.publish()
        self.assertEqual(first['files'], second['files'])
        extra = self.output / 'local-file'
        extra.write_text('keep this too')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(extra.read_text(), 'keep this too')
        extra.unlink()
        edited = self.output / 'Planner.h'
        edited.write_text('user edit')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(edited.read_text(), 'user edit')
        self.assertFalse(list(self.folder.glob('.build-public-stage-*')))


if __name__ == '__main__':
    unittest.main()
