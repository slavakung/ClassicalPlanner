"""Apply external execution-context exclusions without modifying raw results."""
import datetime as dt


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('measurement timestamps must include a timezone')
    return parsed


def annotate(rows, context):
    intervals = context.get('exclude_timing_intervals', [])
    result = []
    for original in rows:
        row = dict(original)
        reasons = list(row.get('timing_exclusion_reasons', []))
        if intervals:
            try:
                start, end = timestamp(row['started_utc']), timestamp(row['finished_utc'])
                if end < start:
                    raise ValueError('negative execution interval')
                for interval in intervals:
                    lower = timestamp(interval['start_utc'])
                    upper = timestamp(interval['end_utc']) if interval.get('end_utc') else dt.datetime.max.replace(tzinfo=dt.timezone.utc)
                    if start < upper and end > lower:
                        reasons.append(interval['reason'])
            except (KeyError, ValueError, TypeError):
                reasons.append('execution interval missing or invalid; timing context cannot be verified')
        row['timing_exclusion_reasons'] = sorted(set(reasons))
        row['timing_eligible'] = row.get('timing_eligible', True) and not reasons
        result.append(row)
    return result
