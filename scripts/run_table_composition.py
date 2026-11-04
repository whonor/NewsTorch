#!/usr/bin/env python3
"""Run the six pairings from the revised modular-composition table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The shared environment contains a pt24 scatter wheel beside PyTorch 2.5.
# Keep the compatible extension isolated to this study.
sys.path.insert(0, str(ROOT / 'artifacts/table_composition/vendor'))
from experiments.table_composition.components import install
install()
from experiments.sota_composition.evaluation import Composition, run_lifecycle_matrix
from scripts.run_sota_composition import _jobs

STUDY = ROOT / 'experiments/table_composition'
OUTPUT = ROOT / 'artifacts/table_composition'
PAIRS = [('naml', 'sue'), ('nrms', 'sue'), ('lstur', 'sue'),
         ('cne', 'naml'), ('cne', 'nrms'), ('cne', 'lstur')]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def protocol():
    # Preserve the existing study's data, model capacity, and optimizer settings.
    source = ROOT / 'experiments/cne_sue_composition/protocol.yaml'
    value = yaml.safe_load(source.read_text())
    value.update(yaml.safe_load((STUDY / 'protocol.yaml').read_text()))
    value['_compositions'] = tuple(
        Composition('C%d' % (index + 1), 'encoder_substitution', news, user,
                    'none', value['training']['strategy'], (), True)
        for index, (news, user) in enumerate(PAIRS)
    )
    value['_resolved_assets'] = {key: str(ROOT / path)
                                 for key, path in value['assets'].items()}
    return value


def jobs():
    result = _jobs(protocol(), OUTPUT / 'results')
    for job in result:
        command = job['command']
        command[1:2] = [str(Path(__file__).resolve()), 'train']
        # Avoid depending on a logging account for batch experiments.
        job['settings']['wandb'] = 'disabled'
    return result


def report(allow_incomplete=False):
    lifecycle = json.loads((OUTPUT / 'lifecycle.json').read_text())
    checks = {row['composition_id']: row for row in lifecycle['results']}
    rows = []
    for index, (news, user) in enumerate(PAIRS, 1):
        cid = 'C%d' % index
        results = []
        for seed in protocol()['seeds']:
            path = OUTPUT / 'results' / cid / ('seed-%d.json' % seed)
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            metadata = payload['composition']
            if (payload['seed'], metadata['id'], metadata['news_encoder'],
                metadata['user_encoder'], metadata['training']) != (
                    seed, cid, news, user, protocol()['training']['strategy']):
                raise ValueError('Result metadata mismatch: %s' % path)
            results.append(payload['metrics'])
        if len(results) != 5 and not allow_incomplete:
            raise ValueError('%s has %d/5 results' % (cid, len(results)))
        row = dict(id=cid, news_encoder=news.upper(), user_encoder=user.upper(),
                   runs=len(results), lifecycle='%d/5' % checks[cid]['passed'])
        for metric in ('auc', 'mrr', 'ndcg5', 'ndcg10'):
            numbers = [float(item[metric]) for item in results]
            row[metric] = statistics.mean(numbers) if len(numbers) == 5 else None
            row[metric + '_stdev'] = statistics.stdev(numbers) if len(numbers) == 5 else None
        rows.append(row)
    target = OUTPUT / 'report'
    write_json(target / 'summary.json', rows)
    with (target / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {metric: max((row[metric] for row in rows if row[metric] is not None),
                        default=None) for metric in ('auc', 'mrr')}
    lines = []
    for row in rows:
        values = []
        for metric in ('auc', 'mrr'):
            number = row[metric]
            value = '[pending]' if number is None else '%.4f' % number
            if number is not None and number == best[metric]:
                value = r'\textbf{' + value + '}'
            values.append(value)
        lines.append(' & %s & %s & Yes & 0 & %s & %s & %s \\\\' % (
            row['news_encoder'], row['user_encoder'], row['lifecycle'], *values))
    (target / 'table_rows.tex').write_text('\n'.join(lines) + '\n')
    print('Completed %d/30 runs' % sum(row['runs'] for row in rows))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv = [str(ROOT / 'general_runner.py')] + sys.argv[2:]
        runpy.run_path(str(ROOT / 'general_runner.py'), run_name='__main__')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--task', type=int, choices=range(30))
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        settings = dict(protocol().get('model_settings', {}))
        settings['long_term_masking_probability'] = 0.0  # CPU lifecycle fixture
        result = run_lifecycle_matrix(protocol()['_compositions'], settings)
        write_json(OUTPUT / 'lifecycle.json', result)
        if not result['all_passed']:
            raise RuntimeError('Lifecycle failure: see lifecycle.json')
        selected_jobs = jobs()
        for job in selected_jobs:
            job['settings_path'].parent.mkdir(parents=True, exist_ok=True)
            job['settings_path'].write_text(yaml.safe_dump(job['settings']))
        files = [ROOT / 'general_runner.py', ROOT / 'config.py',
                 ROOT / 'base_trainer.py', ROOT / 'utils/_evaluation.py',
                 ROOT / 'dataset_corpus_preprocessing/MIND_corpus_main.py',
                 ROOT / 'models/SOTAComposition.py',
                 ROOT / 'models/modules/newsEncoders.py',
                 ROOT / 'models/modules/userEncoders.py',
                 ROOT / 'experiments/table_composition/components.py',
                 STUDY / 'protocol.yaml', Path(__file__).resolve()]
        write_json(OUTPUT / 'freeze_manifest.json', {
            'protocol': yaml.safe_load((STUDY / 'protocol.yaml').read_text()),
            'expected_runs': 30,
            'compositions': [asdict(item) for item in protocol()['_compositions']],
            'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in files},
            'core_files_edited_by_study': [],
            'adapter_file': 'experiments/table_composition/components.py',
            'runtime': {'torch': __import__('torch').__version__,
                        'torch_scatter': __import__('torch_scatter').__version__},
            'note': 'Zero core edits; study-local registration/adapters are additional infrastructure.',
        })
        for row in result['results']:
            print('%s: %d/5' % (row['composition_id'], row['passed']))
    elif args.command == 'run':
        selected = jobs()
        if args.task is not None:
            selected = [selected[args.task]]
        manifest = json.loads((OUTPUT / 'freeze_manifest.json').read_text())
        for filename, expected in manifest['sha256'].items():
            if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen source changed: ' + filename)
        for job in selected:
            if job['result_path'].exists():
                print('Preserving ' + str(job['result_path']))
                continue
            print('Running %s seed %d' % (job['composition'].id, job['seed']), flush=True)
            subprocess.run(job['command'], cwd=ROOT, check=True)
    else:
        report(args.allow_incomplete)


if __name__ == '__main__':
    main()
