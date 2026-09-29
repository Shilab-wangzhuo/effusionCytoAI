#!/usr/bin/env python3
"""Join cancer-inference scores to one cohort's clinical metadata."""
import argparse
import csv
from pathlib import Path
try:
    from .clinical_weighting import CLASSES, read_samples, write_csv
except ImportError:  # Also support direct execution from this directory.
    from clinical_weighting import CLASSES, read_samples, write_csv


def indexed(path, key):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if key not in fields or len(fields) != len(set(fields)):
            raise ValueError('Invalid header: '+str(path))
        rows = list(reader)
    result = {}
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed row: '+str(path))
        r = {k: v.strip() for k, v in row.items()}
        if not r[key] or r[key] in result:
            raise ValueError('Blank or duplicate ID: '+str(path))
        result[r[key]] = r
    return result


def _summary_schema(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        fields = csv.DictReader(f).fieldnames or []
    if 'sample_id' in fields:
        return 'sample', 'sample_id'
    if 'patient_id' in fields:
        return 'patient', 'patient_id'
    raise ValueError('Summary must contain sample_id or patient_id: ' + str(path))


def _scores_and_cell_count(row, schema, sample_id):
    if schema == 'sample':
        if row.get('status') != 'ok':
            raise ValueError('Unevaluable inference: ' + sample_id)
        n_cells = int(row['n_cells_all'])
        scores = {c: float(row[c + '_mean_probability_all']) for c in CLASSES}
    else:
        # cancer_infer.py writes class counts and mean softmax scores by patient.
        n_cells = sum(int(row[c]) for c in CLASSES)
        scores = {c: float(row['prob_' + c]) for c in CLASSES}
    if n_cells <= 0:
        raise ValueError('No available post-filtering cells: ' + sample_id)
    return n_cells, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True,
                   help='One cohort only; sample,patient_id,reference_class,sex,effusion_type,eligible')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError('Output already exists')
        schema, summary_key = _summary_schema(a.summary)
        summary, meta = indexed(a.summary, summary_key), indexed(a.metadata, 'sample')
        rows = []
        for sid, m in meta.items():
            if m['eligible'].lower() not in ('yes', 'no'):
                raise ValueError('eligible must be yes/no: '+sid)
            if m['eligible'].lower() == 'no':
                continue
            summary_id = sid if schema == 'sample' else m['patient_id']
            if summary_id not in summary:
                raise ValueError('Missing inference: '+sid+'; review/exclude documented zero-cell cases explicitly')
            r = summary[summary_id]
            n_cells, scores = _scores_and_cell_count(r, schema, sid)
            if r.get('reference_class') and r['reference_class'] != m['reference_class']:
                raise ValueError('Conflicting reference: '+sid)
            rows.append(dict(sample_id=sid, patient_id=m['patient_id'], reference_class=m['reference_class'],
                             sex=m['sex'], effusion_type=m['effusion_type'], n_cells=n_cells,
                             **scores))
        if not rows:
            raise ValueError('No eligible rows')
        # Validate before delivering a file, using a temporary file in the destination directory.
        import tempfile
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=a.output.parent) as temp:
            candidate = Path(temp)/'candidate.csv'
            write_csv(candidate, rows)
            read_samples(candidate)
            with a.output.open('x', encoding='utf-8', newline='') as f:
                f.write(candidate.read_text(encoding='utf-8'))
    except (ValueError, KeyError, OSError) as exc:
        p.exit(2, 'Error: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
