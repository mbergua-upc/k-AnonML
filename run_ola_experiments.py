#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs the OLA anonymizer for every dataset, classifier, suppression limit (s) and
optimization metric, storing everything in results/ola_experiments/<metric>/.
    - Datasets: adult, cahousing, cmc, mgm
    - Classifiers: knn, rf, svm, xgb
    - Suppression (s, % of records): 0, 3, 6, 9
    - OLA metrics: aecs, dm, gweight, prec

How it saves work compared with calling baseline_with_repetitions.py once per
metric (which is what run_all_experiments.py does for a single metric):
  * The OLA lattice search only depends on (dataset, k, s). The metric is applied
    afterwards, only to choose one of the minimal k-anonymous nodes it returns
    (`--metric none` in OLA returns all of those nodes, and every metric picks one
    of them). So the search runs once and the best node of each metric is chosen
    from the same result, with the same selection rules as elemam/main.py.
  * The anonymized tables of the (usually 1-4) different chosen nodes are the only
    ones generated. They do not depend on the classifier, so they are shared by
    the 4 classifiers.
  * The k-anonymity checker (precomputed generalization values) is built once per
    (dataset, s) instead of once per k.
  * The tables are written in the cache layout of baseline_with_repetitions.py,
    which then only trains and evaluates the classifiers (one process per
    dataset, classifier and metric, covering all the s values).
  * Anonymization (one process per dataset and s) and classification run in
    parallel; the classifiers of a dataset start as soon as its tables exist.
  * Finished work is skipped, so an interrupted run can be resumed.

Outputs (METRIC in aecs, dm, gweight, prec):
    results/ola_experiments/<METRIC>/<dataset>/ola/anon_tables/metric-<METRIC>/   anonymized tables, anon_timing.csv
    results/ola_experiments/<METRIC>/<dataset>/ola/<classifier>/                  classification metrics per s
    results/ola_experiments/ola_selected_nodes.csv                                 chosen node per dataset, s, k and metric
    results/ola_experiments/logs/                                                  log of every classification run
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'baseline_with_repetitions.py'
OUTPUT_DIR = ROOT / 'results' / 'ola_experiments'
LOGS_DIR = OUTPUT_DIR / 'logs'
SELECTED_NODES_PATH = OUTPUT_DIR / 'ola_selected_nodes.csv'

# DATASETS = ['adult', 'cahousing', 'cmc', 'mgm']

# Aldult dataset is the only one that has been tested with OLA (fig), so it is the only one included in this script.
DATASETS = ['adult']
CLASSIFIERS = ['knn', 'rf', 'svm', 'xgb']
METRICS = ['aecs', 'dm', 'gweight', 'prec']
S_VALUES = [0, 3, 6, 9]

# Quasi-identifier columns of every dataset (same as baseline_with_repetitions.py)
QI_INDEX = {
    'adult': [1, 2, 3, 4, 5, 6, 7, 8],
    'cahousing': [1, 2, 3, 8, 9],
    'cmc': [1, 2, 4],
    'mgm': [1, 2, 3, 4, 5],
}


def metric_dir(metric):
    return OUTPUT_DIR / metric


def anon_folder(dataset, metric):
    # same folder name that baseline_with_repetitions.py builds for the OLA metric
    return metric_dir(metric) / dataset / 'ola' / 'anon_tables' / f'metric-{metric}'


def table_path(dataset, metric, s, k):
    return anon_folder(dataset, metric) / f's_{s}' / f'{dataset}_anonymized_{k}_0.csv'


def result_path(dataset, classifier, metric, s, stop_k):
    return metric_dir(metric) / dataset / 'ola' / classifier / f'{dataset}_ola_{classifier}_k_{stop_k}_s_{s}.csv'


def select_best_node(min_k, metric):
    # same rules as elemam/main.py: the first node with the best value wins
    best = None
    best_value = None
    for node in min_k:
        if metric == 'prec':
            value = node.prec
            better = value < (1 if best_value is None else best_value)
        elif metric == 'gweight':
            value = node.level
            better = best_value is None or value < best_value
        elif metric == 'aecs':
            value = node.eqclasses
            better = value != 0 and value < (9999999 if best_value is None else best_value)
        elif metric == 'dm':
            value = node.DM_penalty
            better = best_value is None or value < best_value
        else:
            raise ValueError(f'Unsupported metric: {metric}')
        if better:
            best, best_value = node, value
    # elemam/main.py would fail when no node qualifies (only the top node exists)
    return best if best is not None else min_k[0]


def anonymize(dataset, s, start_k, stop_k, step_k):
    """Runs the OLA lattice search for every k and writes the table of each metric"""
    from elemam.algorithm import calculate, sort
    from elemam.kanon import AnonCheck
    from elemam.main import create_anon_data
    from elemam.node import Node
    from generalization.generalization import age, hierarchy, l1sub
    from utils.data import write_anon

    os.chdir(ROOT)
    gen_path = os.path.join('generalization', 'hierarchies', dataset, '')
    with open(os.path.join('datasets', dataset, f'{dataset}.csv'), newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        header = next(reader)
        rows = list(reader)

    qi_index = QI_INDEX[dataset]
    qi_names = [header[i] for i in qi_index]
    raw_data = [list(column) for column in zip(*rows)]
    qi_data = [raw_data[i] for i in qi_index]

    if dataset == 'adult':
        max_gen_level = [1, 4, 1, 2, 3, 2, 2, 2]
        gen_strat = [
            l1sub, age, l1sub,
            hierarchy(gen_path + dataset, 'marital-status'),
            hierarchy(gen_path + dataset, 'education'),
            hierarchy(gen_path + dataset, 'native-country'),
            hierarchy(gen_path + dataset, 'workclass'),
            hierarchy(gen_path + dataset, 'occupation'),
        ]
    else:
        gen_strat = [hierarchy(gen_path + dataset, name) for name in qi_names]
        max_gen_level = [len(strat[1]) for strat in gen_strat]

    allowed_suppressed = int(len(rows) * (s / 100))
    log_folder = LOGS_DIR / f'anonymization_{dataset}_s_{s}'
    log_folder.mkdir(parents=True, exist_ok=True)

    # the precomputed generalization values only depend on the data, so one checker serves every k
    checker = AnonCheck(qi_data, max_gen_level, gen_strat, allowed_suppressed, 2)

    selected, timings = [], []
    for k in range(start_k, stop_k + 1, step_k):
        if all(table_path(dataset, m, s, k).is_file() for m in METRICS):
            continue
        start_time = time.perf_counter()

        checker.k = k
        checker.eq_classes_dict = None
        checker.prev_gen_to_apply = None
        checker.is_transformed = True

        node_array, level_nodes = [], {}
        Node([0] * len(qi_index), node_array, max_gen_level, level_nodes)
        min_k = calculate(sort(node_array), checker)

        chosen = {m: select_best_node(min_k, m) for m in METRICS}
        unique_nodes = list({tuple(n.attributes): n for n in chosen.values()}.values())
        with redirect_stdout(io.StringIO()):
            tables = create_anon_data(unique_nodes, raw_data, qi_index, gen_strat, k, str(log_folder))
        seconds = time.perf_counter() - start_time

        for metric, node in chosen.items():
            s_folder = anon_folder(dataset, metric) / f's_{s}'
            s_folder.mkdir(parents=True, exist_ok=True)
            write_anon(str(s_folder), tables[tuple(node.attributes)], header, k, s, dataset)
            selected.append((dataset, s, k, metric, ' '.join(map(str, node.attributes)), len(min_k)))
            # the search and node generation are shared by all the metrics
            timings.append((metric, s, k, round(seconds, 3)))
    return dataset, s, selected, timings


def classify(dataset, classifier, metric, start_k, stop_k, step_k):
    """Trains and evaluates the classifier on the cached tables of one metric for all s values"""
    if all(result_path(dataset, classifier, metric, s, stop_k).is_file() for s in S_VALUES):
        return dataset, classifier, metric, 0, True
    cmd = [
        sys.executable, str(SCRIPT),
        '--dataset', dataset,
        '--classifier', classifier,
        '--start-k', str(start_k),
        '--stop-k', str(stop_k),
        '--step-k', str(step_k),
        '--results-dir', str(metric_dir(metric)),
        'ola',
        '--start-s', str(min(S_VALUES)),
        '--stop-s', str(max(S_VALUES)),
        '--step-s', str(S_VALUES[1] - S_VALUES[0]),
        '--metric', metric,
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / f'{dataset}_{classifier}_{metric}.log', 'w') as log:
        returncode = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return dataset, classifier, metric, returncode, False


def append_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def save_anonymization_results(dataset, selected, timings):
    append_rows(SELECTED_NODES_PATH, ['dataset', 's', 'k', 'metric', 'node', 'minimal_nodes'], selected)
    for metric in METRICS:
        rows = [(s, k, seconds, 1) for m, s, k, seconds in timings if m == metric]
        append_rows(anon_folder(dataset, metric) / 'anon_timing.csv',
                    ['s', 'k', 'anonymization_seconds', 'tables'], rows)


def main():
    parser = argparse.ArgumentParser(description='Run OLA for all datasets, classifiers, s values and metrics')
    parser.add_argument('--start-k', type=int, default=2)
    parser.add_argument('--stop-k', type=int, default=100)
    parser.add_argument('--step-k', type=int, default=1)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help='parallel anonymization processes and classification runs')
    args = parser.parse_args()

    if args.start_k < 2 or args.stop_k < args.start_k or args.step_k < 1 \
            or args.start_k + args.step_k > args.stop_k:
        sys.exit('invalid k range')

    k_args = (args.start_k, args.stop_k, args.step_k)
    pending = {dataset: len(S_VALUES) for dataset in DATASETS}
    anon_failed = set()
    failures = []
    classification_futures = []

    with ProcessPoolExecutor(max_workers=min(args.workers, len(DATASETS) * len(S_VALUES))) as anon_pool, \
            ThreadPoolExecutor(max_workers=args.workers) as clf_pool:
        anon_futures = {
            anon_pool.submit(anonymize, dataset, s, *k_args): (dataset, s)
            for s, dataset in product(S_VALUES, DATASETS)
        }
        for future in as_completed(anon_futures):
            dataset, s = anon_futures[future]
            try:
                _, _, selected, timings = future.result()
                save_anonymization_results(dataset, selected, timings)
                print(f'Anonymization finished: {dataset} s={s} ({len(selected) // len(METRICS)} new k values)', flush=True)
            except Exception as error:
                anon_failed.add(dataset)
                failures.append(f'anonymization {dataset} s={s}: {error!r}')
                print(f'FAILED anonymization: {dataset} s={s}: {error!r}', flush=True)
            pending[dataset] -= 1
            if pending[dataset] == 0 and dataset not in anon_failed:
                for classifier, metric in product(CLASSIFIERS, METRICS):
                    classification_futures.append(clf_pool.submit(classify, dataset, classifier, metric, *k_args))

        for future in as_completed(classification_futures):
            dataset, classifier, metric, returncode, skipped = future.result()
            if returncode != 0:
                failures.append(f'classification {dataset} {classifier} {metric} (exit code {returncode})')
                print(f'FAILED (exit code {returncode}): {dataset} {classifier} {metric}', flush=True)
            else:
                print(f'Classification {"already done" if skipped else "finished"}: '
                      f'{dataset} {classifier} {metric}', flush=True)

    if failures:
        print(f'\n{len(failures)} task(s) failed:')
        for failure in failures:
            print(f'  {failure}')
        sys.exit(1)
    print(f'\nAll runs finished successfully. Results in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

# To run this script (with the .venv activado), use the command: python run_ola_experiments.py
