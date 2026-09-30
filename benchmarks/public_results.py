"""Complete publication matrix; no solver, plotting, or live-file mutations."""
from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from measurement_context import annotate

ALGORITHMS = ('bfs dfs recursive-dfs bbdfs ucs astar greedy weighted-astar beam '
              'iddfs idastar rbfs hill ehc backward-bfs bidirectional forward-vi '
              'backward-vi rollout adp adp-astar graphplan satplan regression csp partial-order').split()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_snapshot(folder):
    """Read each source artifact once. A torn last result is visibly omitted."""
    folder = Path(folder)
    campaign_bytes = (folder / 'campaign.json').read_bytes()
    status_bytes = (folder / 'status.json').read_bytes()
    context_path = folder / 'measurement_context.json'
    context_bytes = context_path.read_bytes() if context_path.exists() else b'{}'
    raw_bytes = (folder / 'raw.jsonl').read_bytes() if (folder / 'raw.jsonl').exists() else b''
    rows = []
    tail_omitted = False
    lines = raw_bytes.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('result row must be a JSON object')
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError('corrupt non-final record; refusing to hide missing measurements')
            tail_omitted = True
            break
        rows.append(row)
    return {'campaign': json.loads(campaign_bytes), 'status': json.loads(status_bytes),
            'context': json.loads(context_bytes), 'rows': rows,
            'snapshot': {'captured_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                         'raw_sha256': digest(raw_bytes), 'raw_bytes': len(raw_bytes),
                         'campaign_sha256': digest(campaign_bytes), 'status_sha256': digest(status_bytes),
                         'measurement_context_sha256': digest(context_bytes),
                         'incomplete_trailing_record_omitted': tail_omitted}}


def terminal(campaign, status, now=None):
    now = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
    deadline = campaign.get('deadline_epoch')
    if status.get('status') == 'deadline_reached':
        return number(deadline) and now >= deadline
    if status.get('status') == 'complete':
        return bool(campaign.get('spec', {}).get('rounds')) or (number(deadline) and now >= deadline)
    return False


def summarize(values):
    series = [value for value in values if number(value)]
    if not series:
        return {'n': 0, 'median': None, 'min': None, 'max': None}
    return {'n': len(series), 'median': statistics.median(series), 'min': min(series), 'max': max(series)}


def reference_evidence(instance, campaign, current_hashes):
    """Use bundled documented references only for their exact generated inputs."""
    reference = instance.get('optimal_length')
    reasons = []
    if instance.get('source_kind') != 'generated' or not number(reference) or reference < 0:
        reasons.append('no supported generated-task optimal reference')
    if not instance.get('optimality_basis'):
        reasons.append('reference has no documented proof or exhaustive-search basis')
    original = next((case.get('instance', {}) for case in campaign.get('spec', {}).get('cases', [])
                     if case.get('instance', {}).get('id') == instance['id']), None)
    if original is None or any(original.get(key) != instance.get(key) for key in ('optimal_length', 'optimality_basis')):
        reasons.append('documented reference differs from the original captured case metadata')
    captured = campaign.get('spec', {}).get('inputs_sha256', {})
    matching_inputs = True
    for field in ('domain', 'problem'):
        name = instance[field]
        expected = instance.get(field + '_sha256')
        if not expected or captured.get(name) != expected or current_hashes.get(name) != expected:
            reasons.append(field + ' hash does not match manifest, campaign and distributed input')
            matching_inputs = False
    return {'valid': not reasons, 'optimal_cost': reference if not reasons else None,
            'input_hashes_match': matching_inputs,
            'objective': 'unit_action_cost', 'basis': instance.get('optimality_basis'),
            'reference': instance.get('reference'), 'reasons': reasons,
            'scope': 'bundled documented reference on hash-matched generated inputs; not an algorithm-wide guarantee'}


def proof(row, instance, reference, campaign):
    measurement = row.get('measurement', {})
    reasons = []
    if row.get('status') != 'solved' or measurement.get('status') != 'solved':
        reasons.append('no reported solution')
    if measurement.get('replay_valid') is not True:
        reasons.append('independent original grounded-IR replay is not valid')
    if row.get('spec_sha256') != campaign.get('spec_sha256'):
        reasons.append('measurement protocol/build/input provenance does not match campaign')
    if row.get('instance_id') != instance['id']:
        reasons.append('instance identity mismatch')
    if not reference['input_hashes_match']:
        reasons.append('distributed inputs do not match the originally replayed campaign inputs')
    if measurement.get('algorithm') != row.get('algorithm'):
        reasons.append('measurement algorithm does not match recorded algorithm')
    cost, length, actions = measurement.get('plan_cost'), measurement.get('plan_length'), measurement.get('plan_actions')
    if not number(cost) or not number(length) or length < 0 or not isinstance(actions, list) or len(actions) != length:
        reasons.append('complete plan, original cost or length is missing/inconsistent')
    valid = not reasons
    optimality = 'not_proven'
    if valid and reference['valid'] and measurement.get('unit_action_costs') is True:
        if not math.isclose(cost, length, rel_tol=1e-9, abs_tol=1e-9):
            optimality = 'reference_conflict'
        elif math.isclose(cost, reference['optimal_cost'], rel_tol=1e-9, abs_tol=1e-9):
            optimality = 'proven_reference_match'
        elif cost > reference['optimal_cost']:
            optimality = 'proven_suboptimal_against_reference'
        else:
            optimality = 'reference_conflict'
    return {'feasibility': 'feasible' if valid else 'not_proven', 'optimality': optimality,
            'replayed_complete_plan': valid, 'reasons': reasons,
            'evidence': 'recorded independent original grounded-IR replay; no new planner run during export'}


def build_matrix(manifest, snapshot, current_hashes):
    campaign, state = snapshot['campaign'], snapshot['status']
    instances = manifest['instances']
    instance_by_id = {value['id']: value for value in instances}
    if len(instance_by_id) != len(instances):
        raise ValueError('duplicate instance IDs')
    # Audit all attempts; aggregate only the latest attempt for each round/case ID.
    raw_rows = annotate(snapshot['rows'], snapshot['context'])
    latest = {}
    unknown = []
    for row in raw_rows:
        if row.get('instance_id') not in instance_by_id or row.get('algorithm') not in ALGORITHMS:
            unknown.append(row.get('job_id'))
            continue
        latest[row['job_id']] = row
    groups = defaultdict(list)
    for row in latest.values():
        groups[row['algorithm'], row['instance_id']].append(row)
    active = None
    if state.get('status') == 'running' and state.get('active_job') and state['active_job'] not in latest:
        cases = campaign.get('spec', {}).get('cases', [])
        try:
            case = cases[int(state['active_job'].split(':')[1])]
            active = (case['algorithm'], case['instance']['id'])
        except (ValueError, IndexError, KeyError):
            pass
    finished = terminal(campaign, state) and not snapshot['snapshot']['incomplete_trailing_record_omitted']
    window_complete = finished and number(campaign.get('deadline_epoch')) and dt.datetime.now(dt.timezone.utc).timestamp() >= campaign['deadline_epoch']
    matrix = []
    references = {instance['id']: reference_evidence(instance, campaign, current_hashes) for instance in instances}
    for algorithm in ALGORITHMS:
        for instance in instances:
            rows = sorted(groups[algorithm, instance['id']], key=lambda row: (row.get('round', -1), row['job_id']))
            reference = references[instance['id']]
            evidence = [(row, proof(row, instance, reference, campaign)) for row in rows]
            valid = [row for row, value in evidence if value['replayed_complete_plan']]
            eligible = [row for row in valid if row['timing_eligible']]
            statuses = Counter('invalid_plan' if row.get('status') == 'solved' and not value['replayed_complete_plan']
                               else 'bounded_unsolved' if row.get('status') == 'unsolved' else row.get('status', 'error')
                               for row, value in evidence)
            optimalities = Counter(value['optimality'] for _, value in evidence)
            is_active = active == (algorithm, instance['id'])
            if valid:
                outcome = 'solved'
            elif is_active:
                outcome = 'pending'
            elif not rows:
                outcome = 'not_attempted'
            elif len(statuses) == 1:
                outcome = next(iter(statuses))
            else:
                outcome = 'mixed'
            matched = optimalities['proven_reference_match']
            suboptimal = optimalities['proven_suboptimal_against_reference']
            if optimalities['reference_conflict']:
                optimality = 'reference_conflict'
            elif matched:
                optimality = 'proven_reference_match_all_solutions' if matched == len(valid) else 'proven_reference_match_some_solutions'
            elif suboptimal:
                optimality = 'proven_suboptimal_solutions' if suboptimal == len(valid) else 'some_proven_suboptimal_solutions'
            else:
                optimality = 'not_proven'
            timing = {}
            for key in ('planning_seconds', 'parse_seconds', 'grounding_seconds', 'replay_seconds'):
                timing[key] = summarize(row['measurement'].get(key) for row in eligible)
            timing['wall_seconds'] = summarize(row.get('wall_seconds') for row in eligible)
            by_outcome = {}
            for status in sorted(statuses):
                selected = [row for row, value in evidence
                            if ('invalid_plan' if row.get('status') == 'solved' and not value['replayed_complete_plan']
                                else 'bounded_unsolved' if row.get('status') == 'unsolved' else row.get('status', 'error')) == status]
                by_outcome[status] = {'attempts': len(selected),
                                      'eligible_process_wall_seconds': summarize(row.get('wall_seconds') for row in selected if row['timing_eligible']),
                                      'censored': status in ('timeout', 'deadline', 'interrupted', 'resource_exhausted')}
            matrix.append({'algorithm': algorithm, 'instance_id': instance['id'], 'family': instance['family'],
                           'representation': instance.get('representation'), 'outcome': outcome,
                           'active_job': state.get('active_job') if is_active else None,
                           'attempted': bool(rows), 'recorded_repetitions': len(rows),
                           'rounds': [row.get('round') for row in rows], 'status_counts': dict(statuses),
                           'feasibility': 'feasible' if valid else 'not_proven',
                           'infeasibility': 'not_proven',
                           'feasibility_note': 'A valid full-plan replay proves feasibility; bounded no-plan, timeout, resource exhaustion and errors do not prove infeasibility.',
                           'optimality': optimality, 'optimality_counts': dict(optimalities),
                           'reference_evidence': reference,
                           'valid_replayed_solutions': len(valid), 'reference_matching_solutions': matched,
                           'best_plan_cost': min((row['measurement']['plan_cost'] for row in valid), default=None),
                           'plan_cost': summarize(row['measurement']['plan_cost'] for row in valid),
                           'plan_length': summarize(row['measurement']['plan_length'] for row in valid),
                           'eligible_timing_samples': len(eligible), 'excluded_solved_timing_samples': len(valid) - len(eligible),
                           'timing_status': 'eligible_samples_available' if eligible else 'no_eligible_solved_samples',
                           'timing': timing, 'timing_by_outcome': by_outcome,
                           'timing_exclusion_reasons': sorted({reason for row in rows for reason in row['timing_exclusion_reasons']}),
                           'configured_timeout_seconds': campaign.get('spec', {}).get('timeout_seconds'),
                           'configured_memory_gib': campaign.get('spec', {}).get('memory_gib'),
                           'job_ids': [row['job_id'] for row in rows],
                           'proof_failures': [{'job_id': row['job_id'], 'reasons': value['reasons']}
                                              for row, value in evidence if row.get('status') == 'solved' and not value['replayed_complete_plan']]})
    summary = {'schema_version': 1, 'publication_state': 'final' if window_complete else 'preliminary',
               'campaign_finished': finished, 'campaign_status': state.get('status'),
               'campaign_window_complete': window_complete,
               'configured_duration_days': campaign.get('spec', {}).get('duration_days'),
               'started_utc': campaign.get('started_utc'), 'deadline_utc': campaign.get('deadline_utc'),
               'snapshot': snapshot['snapshot'], 'algorithm_count': len(ALGORITHMS), 'instance_count': len(instances),
               'expected_cases': len(ALGORITHMS) * len(instances), 'matrix_cases': len(matrix),
               'raw_attempts': len(raw_rows), 'recorded_repetitions': len(latest),
               'superseded_attempts': len(raw_rows) - len(latest) - len(unknown), 'unknown_job_ids': unknown,
               'attempted_cases': sum(row['attempted'] for row in matrix),
               'cases_with_valid_solution': sum(row['valid_replayed_solutions'] > 0 for row in matrix),
               'cases_with_reference_matching_solution': sum(row['reference_matching_solutions'] > 0 for row in matrix),
               'cases_with_eligible_timing': sum(row['eligible_timing_samples'] > 0 for row in matrix),
               'pending_cases': sum(row['active_job'] is not None for row in matrix),
               'outcome_counts': dict(Counter(row['outcome'] for row in matrix)),
               'timing_note': 'Only replay-valid complete solutions allowed by captured measurement_context enter solve-time statistics. Exclusion does not erase correctness evidence.',
               'proof_note': 'Optimality means a returned solution matches a documented generated-task optimum with matching input hashes and unit costs. No blanket algorithm guarantee or infeasibility conclusion is inferred.',
               'measurement_context': snapshot['context']}
    return {'summary': summary, 'algorithms': ALGORITHMS, 'instances': instances, 'cases': matrix}, raw_rows
