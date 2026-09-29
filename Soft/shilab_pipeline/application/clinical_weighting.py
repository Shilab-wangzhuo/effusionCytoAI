#!/usr/bin/env python3
"""Select clinical score weights on discovery data and evaluate locked weights.

Python 3.9+, standard library only. Research use, not clinical decision support.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

CLASSES = ('Gastrointestinal_Breast', 'Gynecologic', 'Lung', 'Mesothelioma')
MAJOR = CLASSES[:3]
LUNG_GRID = (1., .5, .25, .1, .05)
GYN_GRID = (1., .7, .5, .3)
VERSION = '1.0.0'


@dataclass(frozen=True)
class Sample:
    sample_id: str
    patient_id: str
    reference: str
    sex: str
    site: str
    scores: tuple
    n_cells: int


@dataclass(frozen=True)
class Rule:
    male_gyn: int = 1
    ascites_lung: float = 1.
    pleural_gyn: float = 1.

    def as_dict(self):
        return dict(male_gyn=self.male_gyn, ascites_lung=self.ascites_lung,
                    pleural_gyn=self.pleural_gyn)


def predict(s, rule):
    weights = [1., 1., 1., 1.]
    if s.sex == 'Male':
        weights[1] *= rule.male_gyn
    if s.site == 'Ascite':
        weights[2] *= rule.ascites_lung
    if s.site == 'PE':
        weights[1] *= rule.pleural_gyn
    scores = [a*b for a, b in zip(s.scores, weights)]
    total = math.fsum(scores)
    if total <= 0:
        return (), (None,)*4
    scores = tuple(x/total for x in scores)
    ranking = tuple(CLASSES[i] for i in sorted(range(4), key=lambda i: (-scores[i], i))
                    if weights[i] > 0)
    return ranking, scores


def read_samples(path):
    """One evaluable specimen per patient; do not silently drop missing data."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = ['sample_id', 'patient_id', 'reference_class', 'sex',
                    'effusion_type', 'n_cells'] + list(CLASSES)
        if len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError('Duplicate headers or missing columns: ' + str(sorted(set(required)-set(fields))))
        rows = list(reader)
    result = []
    for r in rows:
        if None in r or any(v is None for v in r.values()):
            raise ValueError('Malformed CSV row')
        r = {k: v.strip() for k, v in r.items()}
        sid, pid = r['sample_id'], r['patient_id']
        sex = {'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'}.get(r['sex'])
        site = {'PE': 'PE', 'Ascite': 'Ascite', 'Ascites': 'Ascite'}.get(r['effusion_type'])
        scores = tuple(float(r[c]) for c in CLASSES)
        n = int(r['n_cells'])
        if not sid or not pid or not sex or not site or r['reference_class'] not in CLASSES:
            raise ValueError('Invalid identifiers, metadata or reference class: '+sid)
        if n <= 0:
            raise ValueError('No usable cells: '+sid+'; exclude and report separately before analysis')
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError('Invalid score: '+sid)
        if not math.isclose(math.fsum(scores), 1, rel_tol=0, abs_tol=1e-5):
            raise ValueError('Scores must sum to 1 within 1e-5: '+sid)
        if sex == 'Male' and r['reference_class'] == 'Gynecologic':
            raise ValueError('Reference conflicts with male GYN exclusion; review metadata/class definition: '+sid)
        result.append(Sample(sid, pid, r['reference_class'], sex, site, scores, n))
    if not result:
        raise ValueError('Empty input')
    for key in ('sample_id', 'patient_id'):
        if len({getattr(s, key) for s in result}) != len(result):
            raise ValueError('Duplicate '+key+'; this version requires one specimen per patient')
    return sorted(result, key=lambda s: s.sample_id)


def all_rules():
    return [Rule(*v) for v in itertools.product((1, 0), LUNG_GRID, GYN_GRID)]


def selection_key(samples, rule):
    support = Counter(s.reference for s in samples)
    if any(not support[c] for c in MAJOR):
        raise ValueError('All three major classes required for selection')
    tp = Counter(s.reference for s in samples if predict(s, rule)[0][:1] == (s.reference,))
    recall = sum((Fraction(tp[c], support[c]) for c in MAJOR), Fraction())/3
    accuracy = Fraction(sum(tp.values()), len(samples))
    penalty = -math.log(rule.ascites_lung)-math.log(rule.pleural_gyn)
    return recall, accuracy, -penalty, rule.ascites_lung, rule.pleural_gyn, rule.male_gyn


def choose_rule(samples):
    return max((r for r in all_rules() if r.male_gyn == 0), key=lambda r: selection_key(samples, r))


def make_folds(samples, seed):
    # Retains the historical deterministic stratified five-fold allocation.
    # Mesothelioma is supported when present, without requiring five such cases.
    rng, assignment, offset = random.Random(seed), {}, 0
    for c in CLASSES:
        group = sorted((s for s in samples if s.reference == c), key=lambda s: s.patient_id)
        if c in MAJOR and len(group) < 5:
            raise ValueError(c+' requires at least 5 patients for five-fold CV')
        rng.shuffle(group)
        for i, s in enumerate(group):
            assignment[s.sample_id] = (i+offset) % 5+1
        offset = (offset+len(group)) % 5
    return assignment


def cross_validate(samples, seed):
    folds, predictions, chosen = make_folds(samples, seed), {}, []
    for fold in range(1, 6):
        training = [s for s in samples if folds[s.sample_id] != fold]
        held = [s for s in samples if folds[s.sample_id] == fold]
        rule = choose_rule(training)
        chosen.append(dict(fold=fold, training_n=len(training), held_out_n=len(held), **rule.as_dict()))
        for s in held:
            predictions[s.sample_id] = predict(s, rule)
    return [predictions[s.sample_id] for s in samples], folds, chosen


def metrics(samples, predictions):
    support = Counter(s.reference for s in samples)
    ranked = [p[0] for p in predictions]
    top = [r[0] if r else 'Unclassifiable' for r in ranked]
    tp = Counter(s.reference for s, c in zip(samples, top) if s.reference == c)
    predicted = Counter(top)
    matrix = Counter((s.reference, c) for s, c in zip(samples, top))
    per_class = []
    for c in CLASSES:
        row = dict(class_name=c, reference_n=support[c], predicted_top1_n=predicted[c])
        for k in (1, 2, 3):
            n = sum(s.reference == c and c in r[:k] for s, r in zip(samples, ranked))
            row.update({f'top{k}_correct_n': n, f'top{k}_accuracy': n/support[c] if support[c] else None})
        row.update(precision=tp[c]/predicted[c] if predicted[c] else None,
                   f1=2*tp[c]/(support[c]+predicted[c]) if support[c]+predicted[c] else None)
        per_class.append(row)
    result = dict(n_samples=len(samples), unclassifiable=predicted['Unclassifiable'],
                  predicted_mesothelioma=predicted['Mesothelioma'])
    for k in (1, 2, 3):
        n = sum(s.reference in r[:k] for s, r in zip(samples, ranked))
        result.update({f'top{k}_correct_n': n, f'top{k}_accuracy': n/len(samples)})
    result['three_class_mean_recall'] = (sum(tp[c]/support[c] for c in MAJOR)/3
                                          if all(support[c] for c in MAJOR) else None)
    old = [predict(s, Rule())[0][0] == s.reference for s in samples]
    new = [s.reference == c for s, c in zip(samples, top)]
    result['wrong_to_correct'] = sum(not a and b for a, b in zip(old, new))
    result['correct_to_wrong'] = sum(a and not b for a, b in zip(old, new))
    return result, per_class, [dict(reference_class=a, predicted_class=b, n=matrix[a,b])
                             for a in CLASSES for b in CLASSES+('Unclassifiable',)]


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def report(out, name, samples, predictions, folds=None):
    summary, per_class, matrix = metrics(samples, predictions)
    write_json(out/(name+'_metrics.json'), summary)
    write_csv(out/(name+'_per_class.csv'), per_class)
    write_csv(out/(name+'_confusion.csv'), matrix)
    rows = []
    for s, (ranking, scores) in zip(samples, predictions):
        r = dict(sample_id=s.sample_id, patient_id=s.patient_id, reference_class=s.reference,
                 sex=s.sex, effusion_type=s.site, n_cells=s.n_cells,
                 fold=folds.get(s.sample_id, '') if folds else '',
                 status='ok' if ranking else 'Unclassifiable')
        r.update({f'top{k}': ranking[k-1] if len(ranking) >= k else '' for k in (1, 2, 3)})
        r.update({c: score for c, score in zip(CLASSES, scores)})
        rows.append(r)
    write_csv(out/(name+'_predictions.csv'), rows)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_evaluation_rule(weights_path, samples):
    """Load either an audited fit output or the project's locked rule file."""
    config = json.loads(Path(weights_path).read_text(encoding='utf-8'))
    if config.get('schema_version') == 1 and config.get('classes') == list(CLASSES):
        for key in ('sample_id', 'patient_id'):
            existing = config.get(key+'_sha256')
            if not isinstance(existing, list) or len(existing) != config.get('discovery_n'):
                raise ValueError('Missing discovery identity audit in config')
            if any(id_digest(getattr(s, key)) in set(existing) for s in samples):
                raise ValueError('Discovery/test overlap detected: '+key)
        rule = Rule(**config['weights'])
        overlap_check = 'sha256 identifiers checked'
    elif isinstance(config.get('selected'), dict):
        selected = config['selected']
        try:
            rule = Rule(
                male_gyn=int(selected['male_gyn_weight']),
                ascites_lung=float(selected['ascites_lung_weight']),
                pleural_gyn=float(selected['pleural_gyn_weight']),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Locked weights file has invalid selected-rule fields') from exc
        expected_id = (
            f'M{rule.male_gyn}_A{rule.ascites_lung:g}_P{rule.pleural_gyn:g}'
        )
        if selected.get('rule_id') != expected_id:
            raise ValueError('Locked weights rule_id does not match its weight values')
        overlap_check = 'not available in locked weights file; verify cohort independence separately'
        print('Warning: this locked weights file contains no discovery patient IDs; '
              'verify that the evaluation cohort is independent of the discovery cohort.')
    else:
        raise ValueError('Unsupported weights file: expected a version-1 fit output or a locked selected-rule config')

    if rule not in [r for r in all_rules() if r.male_gyn == 0]:
        raise ValueError('Weights are not from the specified selection grid')
    return rule, overlap_check


def run(args):
    samples = read_samples(args.input)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty')
    if args.command == 'fit':
        rule = choose_rule(samples)
        oof, folds, choices = cross_validate(samples, args.seed)
        grid = []
        for r in sorted(all_rules(), key=lambda r: selection_key(samples, r), reverse=True):
            m = metrics(samples, [predict(s, r) for s in samples])[0]
            grid.append(dict(**r.as_dict(), selection_candidate=r.male_gyn == 0,
                             selected=r == rule, **m))
        config = dict(schema_version=1, software_version=VERSION, classes=list(CLASSES),
                      weights=rule.as_dict(), seed=args.seed, discovery_n=len(samples),
                      input_sha256=file_digest(args.input),
                      sample_id_sha256=[id_digest(s.sample_id) for s in samples],
                      patient_id_sha256=[id_digest(s.patient_id) for s in samples],
                      selection='three-class mean recall, Top1 accuracy, least adjustment, larger Lung weight, larger GYN weight',
                      caveat='Rules were motivated by discovery errors. Five-fold CV is exploratory, not independent validation.')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'weights.json', config)
        write_csv(out/'grid_search.csv', grid)
        write_csv(out/'cv_weights.csv', choices)
        report(out, 'discovery_oof', samples, oof, folds)
        report(out, 'discovery_apparent', samples, [predict(s, rule) for s in samples])
    else:
        rule, overlap_check = load_evaluation_rule(args.weights, samples)
        predictions = [predict(s, rule) for s in samples]
        out.mkdir(parents=True, exist_ok=True)
        report(out, 'fixed_weight', samples, predictions)
        write_json(out/'evaluation_audit.json', dict(input_sha256=file_digest(args.input),
                   weights_sha256=file_digest(args.weights), weights=rule.as_dict(),
                    software_version=VERSION, identity_overlap_check=overlap_check,
                    note='No weight search on evaluation data'))
    report(out, 'baseline', samples, [predict(s, Rule()) for s in samples])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('fit', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'fit':
            p.add_argument('--seed', type=int, default=42)
        else:
            p.add_argument('--weights', type=Path, required=True)
    try:
        run(parser.parse_args())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, 'Error: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
