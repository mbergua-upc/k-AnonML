#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs baseline_with_repetitions.py for every combination of dataset, classifier
and anonymization method defined below.
    - Datasets: adult, cahousing, cmc, mgm
    - Classifiers: knn, rf, svm, xgb
    - Anonymization methods: mondrian, tdg, cb (Default: knn), ola (Default: suppresion 3%, gweight)
"""

import csv
import subprocess
import sys
from itertools import product
from pathlib import Path

DATASETS = ['mgm']
CLASSIFIERS = ['knn', 'rf', 'svm', 'xgb']
# METHODS = ['mondrian', 'tdg', 'cb', 'ola']
METHODS = ['arx_sample', 'arx_optimal', 'arx_bottom_up']

START_K = 2
STOP_K = 100
STEP_K = 1

CB_ALG = 'knn'

ARX_START_S = 3
ARX_STOP_S = 3
ARX_STEP_S = 1

OLA_START_S = 3
OLA_STOP_S = 3
OLA_STEP_S = 1
OLA_METRIC = 'gweight'

SCRIPT = Path(__file__).resolve().parent / 'baseline_with_repetitions.py'
RESULTS_DIR = SCRIPT.parent / 'results'
BASELINE_DIR = RESULTS_DIR / 'original_baseline'
SUMMARY_PATH = RESULTS_DIR / 'original_baseline_summary.csv'


def build_command(dataset, classifier, method):
    cmd = [
        sys.executable, str(SCRIPT),
        '--dataset', dataset,
        '--classifier', classifier,
        '--start-k', str(START_K),
        '--stop-k', str(STOP_K),
        '--step-k', str(STEP_K),
        method,
    ]
    if method == 'cb':
        cmd += ['--cb-alg', CB_ALG]
    elif method == 'ola':
        cmd += [
            '--start-s', str(OLA_START_S),
            '--stop-s', str(OLA_STOP_S),
            '--step-s', str(OLA_STEP_S),
            '--metric', OLA_METRIC,
        ]
    elif method == 'arx_sample' or method == 'arx_optimal' or method == 'arx_bottom_up':
        cmd += [
            '--start-s', str(ARX_START_S),
            '--stop-s', str(ARX_STOP_S),
            '--step-s', str(ARX_STEP_S),
        ]
    return cmd


def write_original_baseline_summary():
    with open(SUMMARY_PATH, 'w', newline='') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['dataset', 'classifier', 'predictor', 'accuracy', 'precision', 'recall', 'f1_score'])
        for dataset, classifier in product(DATASETS, CLASSIFIERS):
            source = BASELINE_DIR / f'{dataset}_{classifier}.csv'
            if not source.exists():
                print(f'WARNING: missing baseline file for {dataset} {classifier}: {source}', flush=True)
                continue
            with open(source, newline='') as source_file:
                for row in csv.DictReader(source_file):
                    writer.writerow([dataset, classifier, row['predictor'], row['accuracy'],
                                     row['precision'], row['recall'], row['f1_score']])
    print(f'Original-data baseline summary written to {SUMMARY_PATH}', flush=True)


def main():
    failures = []
    for dataset, classifier, method in product(DATASETS, CLASSIFIERS, METHODS):
        cmd = build_command(dataset, classifier, method)
        print(f'Running: {" ".join(cmd)}', flush=True)
        result = subprocess.run(cmd, cwd=SCRIPT.parent)
        if result.returncode != 0:
            failures.append((dataset, classifier, method, result.returncode))
            print(f'FAILED (exit code {result.returncode}): {dataset} {classifier} {method}', flush=True)

    write_original_baseline_summary()

    if failures:
        print(f'\n{len(failures)} run(s) failed:')
        for dataset, classifier, method, code in failures:
            print(f'  {dataset} {classifier} {method} (exit code {code})')
        sys.exit(1)

    print('\nAll runs finished successfully.')


if __name__ == '__main__':
    main()

# To run this script (with the .venv activado), use the command: python run_all_experiments.py