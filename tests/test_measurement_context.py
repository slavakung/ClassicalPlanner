import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'benchmarks'))
from measurement_context import annotate


class MeasurementContextTests(unittest.TestCase):
    context = {'exclude_timing_intervals': [{'start_utc': '2026-09-16T12:50:00Z',
                                           'end_utc': '2026-09-17T12:50:00Z', 'reason': 'research overlap'}]}

    def test_crossing_interval_excludes_timing_without_changing_outcome(self):
        source = {'started_utc': '2026-09-16T14:49:00+02:00',
                  'finished_utc': '2026-09-16T14:51:00+02:00', 'status': 'solved'}
        row = annotate([source], self.context)[0]
        self.assertFalse(row['timing_eligible'])
        self.assertEqual(row['status'], 'solved')
        self.assertNotIn('timing_eligible', source)

    def test_run_starting_at_exclusion_end_is_eligible(self):
        row = {'started_utc': '2026-09-17T12:50:00Z', 'finished_utc': '2026-09-17T12:51:00Z'}
        self.assertTrue(annotate([row], self.context)[0]['timing_eligible'])

    def test_missing_time_fails_closed_and_existing_exclusion_survives(self):
        self.assertFalse(annotate([{}], self.context)[0]['timing_eligible'])
        self.assertFalse(annotate([{'timing_eligible': False}], {})[0]['timing_eligible'])


if __name__ == '__main__':
    unittest.main()
