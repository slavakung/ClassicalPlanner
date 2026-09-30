#!/usr/bin/env python3
"""Export an explicitly labelled, portable snapshot with figures for GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from measurement_context import annotate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--note', required=True, help='Conditions and limitations of this snapshot')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / 'campaign.json').read_text())
    cases = metadata['spec']['cases']
    rows = []
    raw_bytes = (source / 'raw.jsonl').read_bytes()
    lines = raw_bytes.splitlines()
    incomplete_tail = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            incomplete_tail = True
            break
        instance = cases[int(row['job_id'].split(':')[1])]['instance']
        if 'measurement' in row:
            row['measurement']['domain_file'] = instance['domain']
            row['measurement']['problem_file'] = instance['problem']
        # Paths in diagnostic exceptions are presentation-only; hashes are preserved.
        row = json.loads(json.dumps(row).replace(str(ROOT), '<repository>'))
        rows.append(row)
    context_path = source / 'measurement_context.json'
    if context_path.exists():
        context = json.loads(context_path.read_text())
        rows = annotate(rows, context)
        (output / 'measurement_context.json').write_text(json.dumps(context, indent=2) + '\n')
    (output / 'raw.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    metadata['environment']['binary'] = 'build/benchmark_planner'
    metadata['export'] = {'note': args.note, 'raw_source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
                          'captured_bytes': len(raw_bytes), 'exported_records': len(rows),
                          'incomplete_trailing_record_omitted': incomplete_tail,
                          'transformations': ['domain/problem paths restored to manifest-relative paths',
                                              'repository prefix removed from diagnostic text and binary path'],
                          'source_status': json.loads((source / 'status.json').read_text())}
    (output / 'campaign.json').write_text(json.dumps(metadata, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'benchmarks/plot_serial.py'), '--input', str(output / 'raw.jsonl'),
                    '--output', str(output)], check=True)
    report = output / 'README.md'
    report.write_text(report.read_text().replace('# Serial comparison results', '# Serial comparison results\n\n' + args.note, 1))


if __name__ == '__main__':
    main()
