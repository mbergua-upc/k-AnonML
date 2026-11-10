#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anonymization and Machine Learning Evaluation Framework

This script implements an experimental framework to evaluate the impact of
different data anonymization algorithms and privacy configurations on machine
learning performance.

INPUTS:
    - Dataset: Adult, CMC, MGM or CAHousing.
    - Classifier: KNN, Random Forest, SVM or XGBoost.
    - Anonymization method:
        * Mondrian
        * Top-Down Greedy (TDG)
        * Clustering-Based Anonymization
        * OLA
        * ARX (Sample, Optimal or Bottom-Up search)
    - Privacy parameters:
        * k-anonymity (range of k values)
        * Suppression rate (for OLA and ARX) (range of s values)
        * l-diversity and t-closeness (optional, for ARX)
    - Additional method-specific parameters, such as clustering algorithm,
      optimization metric and sensitive attribute.

WORKFLOW:
    1. Load the selected dataset and identify quasi-identifiers, sensitive
       attributes and the target variable.
    2. Preprocess the original dataset and evaluate the selected classifier
       to establish a baseline.
    3. Apply the selected anonymization algorithm for each specified privacy
       configuration.
    4. Preprocess the anonymized datasets, handling suppressed values,
       numerical ranges and categorical attributes.
    5. Train and evaluate the classifier on each anonymized dataset using
       the same 70/30 train-test split and random seed (42).
    6. Calculate and store classification performance metrics.

OUTPUTS:
    - Anonymized datasets in CSV format, generated for each privacy
      configuration.
    - Classification results (accuracy, precision, recall and F1-score).
    - Feature matrix dimensions for each experiment.
    - Results organized into timestamped directories by dataset and
      anonymization method.

The main objective is to compare the predictive utility of anonymized data
under different privacy configurations and analyse the trade-off between
privacy protection and machine learning performance.
"""

# Imports
import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import seaborn as sns
import utils.utility as util
from basic_mondrian.anonymizer import get_result_one
from basic_mondrian.utils.read_adult_data import read_tree
from clustering_based.anonymizer import get_result_one as cb_get_result_one
from elemam.main import main as emain
from generalization.generalization import age, hierarchy, l1sub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from top_down_greedy.anonymizer import tdg_get_result_one
from utils.data import read_raw, write_anon
from utils.types import AnonMethod, Classifier, Dataset, MLRes
from arx_wrapper.anonymizer import get_result_arx


def main(args):
    sns.set_theme()  # set defaults (ORIGINAL: sns.set())
    rnd = 42
    np.random.seed(rnd)

    dataset = args.dataset
    anon_method = args.anon_method
    classifier = args.classifier

    # Global Parameter
    k_range = range(args.start_k, args.stop_k + 1, args.step_k)

    # Suppression (OLA and ARX)
    s_range = [0]
    if anon_method == 'ola' or str(anon_method).startswith('arx'):
        s_range = range(args.start_s, args.stop_s + 1, args.step_s)

    # define necessary paths

    # Data path
    path = os.path.join('datasets', dataset, '')  # trailing /
    # Dataset path
    data_path = os.path.join(path, f'{dataset}.csv')
    # Generalization hierarchies path
    gen_path = os.path.join('generalization', 'hierarchies', dataset, '')  # trailing /
    # folder for all results
    res_folder = os.path.join('results', dataset, anon_method, datetime.now(timezone.utc).isoformat().replace(':', '_'))
    # ORIGINAL: datetime.utcnow()

    # ML results path
    output_path = os.path.join(res_folder, f'{dataset}_{os.path.basename(anon_method)}_{os.path.basename(classifier)}_k_{args.stop_k}.csv')
    # path for anonymized datasets
    anon_folder = os.path.join(res_folder, 'anon_dataset', '')  # trailing /
    # path for pickled numeric values
    numeric_folder = os.path.join(res_folder, 'numeric')
    # save ML features
    features_file = os.path.join(res_folder, 'features.csv')

    # create path needed for results recursively
    os.makedirs(anon_folder)
    os.makedirs(numeric_folder)

    xgb_eval_metric = 'error'

    # reading in the data
    data = pd.read_csv(data_path, delimiter=';')
    print('Original Data: ' + str(data.shape[0]) + ' entries, ' + str(data.shape[1]) + ' attributes')

    ATT_NAMES = list(data.columns)

    if dataset == Dataset.CMC:
        QI_INDEX = [1, 2, 4]
        target_var = 'method'
        IS_CAT2 = [False, True, False]
        max_numeric = {"age": 32.5, "children": 8}
        xgb_eval_metric = 'merror'

    elif dataset == Dataset.MGM:
        QI_INDEX = [1, 2, 3, 4, 5]
        target_var = 'severity'
        IS_CAT2 = [True, False, True, True, True]
        max_numeric = {"age": 50.5}

    elif dataset == Dataset.CAHOUSING:
        QI_INDEX = [1, 2, 3, 8, 9]
        target_var = 'ocean_proximity'
        IS_CAT2 = [False, False, False, False, False]
        max_numeric = {"latitude": 119.33, "longitude": 37.245, "housing_median_age": 32.5,
                       "median_house_value": 257500, "median_income": 5.2035}
        xgb_eval_metric = 'merror'

    elif dataset == Dataset.ADULT:
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        target_var = 'salary-class'
        IS_CAT2 = [True, False, True, True, True, True, True, True]
        max_numeric = {"age": 50.5}

    QI_NAMES = list(np.array(ATT_NAMES)[QI_INDEX])
    IS_CAT = [True] * len(QI_INDEX)
    SA_INDEX = [index for index in range(len(ATT_NAMES)) if index not in QI_INDEX]
    SA_var = [ATT_NAMES[i] for i in SA_INDEX]

    # one hot encoding for all categorical values
    one_hot_original = [col for i, col in enumerate(data[QI_NAMES].columns) if IS_CAT2[i]]
    one_hot_anon = one_hot_original

    if anon_method == AnonMethod.OLA:
        gen_strat = [hierarchy(gen_path + dataset, elem) for elem in QI_NAMES]
        # How often a QI can be generalized
        max_gen_level = [len(elem[1]) for elem in gen_strat]

    # override auto parameters as needed
    if dataset == Dataset.ADULT:
        max_gen_level = [1, 4, 1, 2, 3, 2, 2, 2]
        gen_strat = [
            l1sub, age, l1sub,
            hierarchy(gen_path + dataset, 'marital-status'),
            hierarchy(gen_path + dataset, 'education'),
            hierarchy(gen_path + dataset, 'native-country'),
            hierarchy(gen_path + dataset, 'workclass'),
            hierarchy(gen_path + dataset, 'occupation')
        ]

    elif dataset == Dataset.CAHOUSING:
        SA_var = ['ID', 'ocean_proximity']

    elif dataset == Dataset.CMC:
        SA_var = ['ID', 'method']

    # Experiments on original Data

    # label encoding of the target variable
    data[target_var] = data[target_var].astype('category').cat.codes

    # one hot encoding of categorical variables needed for the classification task
    data2 = pd.get_dummies(data, columns=one_hot_original, drop_first=True)

    # creating the ground truth (target variable) vector and removing target variable and ID from the dataset
    y = data[target_var]
    X = data2.drop(SA_var, axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # split the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)

    # creating a classifer
    clf = util.create_classifier(classifier, args.dataset)
    print(clf)

    # train the model using the training sets
    if classifier == Classifier.XGB:
        clf.fit(X_train, y_train, eval_metric=[xgb_eval_metric], eval_set=[
                (X_train, y_train), (X_test, y_test)], early_stopping_rounds=40)
    else:
        clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    print('Results on original data:')
    print('Training data:')
    print('Accuracy: %f%%' % (util.get_classifier_metrics(y_train, pred_train)[0]))
    print('Test data:')
    print('Accuracy: %f%%' % (util.get_classifier_metrics(y_test, pred_test)[0]))

    if args.verbose > 0:
        util.show_classifier_metrics(y_train, pred_train, y_test, pred_test)

    # calculating the zero-rule baseline
    baseline = util.zero_rule_baseline(y_test)
    print('Zero-Rule baseline: %f%%' % (baseline))

    # Data Anonymization and repeated experiments (with different k)

    with open(features_file, 'a+', newline='') as f_file:
        writer = csv.writer(f_file)
        nodes_count = 1
        raw_data, header = read_raw(path, numeric_folder, dataset, QI_INDEX, IS_CAT)
        ATT_TREES = read_tree(gen_path, numeric_folder, dataset, ATT_NAMES, QI_INDEX, IS_CAT)
        for s in s_range:
            s_folder = os.path.join(anon_folder, 's_' + str(s))
            os.mkdir(s_folder)
            ml_res = MLRes()
            for k in k_range:
                anon_data = None
                if anon_method == AnonMethod.MONDRIAN:
                    anon_data = get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX)
                elif anon_method == AnonMethod.TDG:
                    anon_data = tdg_get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX)
                elif anon_method == AnonMethod.CB:
                    anon_data = cb_get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX, args.cb_alg)

                elif anon_method == AnonMethod.OLA:
                    # Anonymize data with OLA
                    anon_data, gen_level_array = emain(raw_data, k, gen_strat, max_gen_level,
                                                       QI_INDEX, args.metric, res_folder, suppression_rate=s)

                elif anon_method in [AnonMethod.ARX_SAMPLE, AnonMethod.ARX_OPTIMAL, AnonMethod.ARX_BOTTOM_UP]:
                    # 'arx_bottom_up' -> 'BOTTOM_UP'
                    arx_method = str(anon_method).split('_', 1)[1].upper()
                    anon_data = get_result_arx(
                        raw_data, header, k, gen_path, dataset, QI_INDEX, arx_method, s,
                        sensitive=args.sensitive or target_var,
                        l_diversity=args.l_diversity, l_variant=args.l_variant,
                        t_closeness=args.t_closeness
                    )

                # Write anonymized data in csv file
                nodes_count = write_anon(s_folder, anon_data, header, k, s, dataset)

                for node in range(nodes_count):
                    # reading in the anonymized data
                    anon_data = pd.read_csv(os.path.join(
                        s_folder, dataset + "_anonymized_" + str(k) + '_' + str(node) + ".csv"), delimiter=';')
                    print(
                        'K: ' + str(k) + ' S: ' + str(s) + ' Node: ' + str(node) + ' | Anonymized Data: ' +
                        str(anon_data.shape[0]) + ' entries, ' + str(anon_data.shape[1]) + ' attributes'
                    )
                    # we have to sort the data with respect to ID (in case a anonymization algorithm rearranges the entries)
                    anon_data = anon_data.sort_values(by=['ID'])

                    # label encoding of the target variable
                    anon_data[target_var] = anon_data[target_var].astype('category').cat.codes

                    # creating the ground truth (target variable) vector and removing target variable and ID from the dataset
                    y = anon_data[target_var]
                    # object dtype allows replacing strings with numbers (pandas >= 3)
                    X = anon_data.drop(SA_var, axis=1).astype(object)

                    for index_row, row in X.iterrows(): 
                        cat_iter = iter(IS_CAT2)
                        for index_col, col in row.items(): # ORIGINAL: for index_col, col in row.iteritems():
                            # only quasi identifiers
                            if index_col not in QI_NAMES:
                                continue

                            # only non categorical attributes
                            if next(cat_iter):
                                continue

                            # replace suppressed value with highest value of according attribute
                            if col == '*':
                                newval = max_numeric.get(index_col)
                                if newval is None:
                                    print('Err: ' + max_numeric.get(index_col) + "   index: " + index_col)
                                X.at[index_row, index_col] = newval
                                continue

                            try:
                                # check if value is a range e.g. a-c
                                val = col.split('-')
                                if len(val) == 1:
                                    continue
                                if val[0] == "" or val[1] == "":
                                    continue

                                # replace range value with mean
                                newval = (float(val[0]) + float(val[1])) / 2
                                if newval is None:
                                    print('Err: ' + max_numeric.get(index_col) + "   index: " + index_col)
                                X.at[index_row, index_col] = newval
                            except AttributeError:
                                pass

                    # replace all categorical value with numeric values
                    for qi in max_numeric.keys():
                        print(qi)
                        X[qi] = pd.to_numeric(X[qi])

                    # one hot encoding of categorical variables needed for the classification task
                    X = pd.get_dummies(X, columns=one_hot_anon, drop_first=True)

                    if args.verbose > 1:
                        print(X)

                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)

                    # write feature
                    writer.writerow(X.shape)

                    # spliting the dataset into training and testing set
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)

                    # creating a classifer
                    clf = util.create_classifier(classifier, args.dataset)

                    # train the model using the training sets
                    if classifier == Classifier.XGB:
                        clf.fit(
                            X_train, y_train, eval_metric=[xgb_eval_metric], eval_set=[
                                (X_train, y_train), (X_test, y_test)], early_stopping_rounds=40
                        )

                    clf.fit(X_train, y_train)
                    pred_train = clf.predict(X_train)
                    pred_test = clf.predict(X_test)
                    print(set(np.asarray(y_test)))
                    print(set(pred_test))

                    # append accuracty, precision, recall and f1 score to the ML results
                    for i, res in enumerate(util.get_classifier_metrics(np.asarray(y_test), pred_test)):
                        ml_res[i].append(res)

                    print(ml_res.f1_score[-1])

                # For OLA debugging
                if args.debug:
                    util.write_results(s, ml_res, args.anon_method, output_path, num=i)
                    ml_res = MLRes()

            util.write_results(s, ml_res, args.anon_method, output_path)

    if args.verbose > 1:
        for att in ml_res:
            print(att)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Anonymize data utilising different algorithms and analyse the effects of the anonymization on the data'
    )

    parser.add_argument(
        '--dataset',
        # ["adult", "cahousing", "cmc", "mgm"],
        choices=list(Dataset),
        default="adult",
        nargs='?',
        help='the dataset used for anonymization'
    )

    # ["rf", "knn", "svm", "xgb"]
    parser.add_argument('--classifier', choices=list(Classifier),
                        default="knn", nargs='?', help='machine learning classifier')
    parser.add_argument('--start-k', default="2", type=int, help='initial value for k of k-anonymity')
    parser.add_argument('--stop-k', default="100", type=int, help='last value for k of k-anonymity')
    parser.add_argument('--step-k', default="1", type=int, help='step for increasing k of k-anonymity')

    subparsers = parser.add_subparsers(dest='anon_method')
    subparsers.required = True
    parser_mon = subparsers.add_parser(AnonMethod.MONDRIAN.value, help='mondrian anonyization algorithm')

    parser_ola = subparsers.add_parser(AnonMethod.OLA.value, help='ola anonyization algorithm')

    parser_ola.add_argument('--start-s', default="3", type=int, help='initial value for suppression of ola')
    parser_ola.add_argument('--stop-s', default="3", type=int, help='last value for suppression of ola')
    parser_ola.add_argument('--step-s', default="1", type=int, help='step for increasing suppression of ola')
    parser_ola.add_argument("--metric", '-m', choices=['none', 'gweight',
                                                       'prec', 'aecs', 'dm', 'ent'], default='gweight', help='ola metric')

    parser_tdg = subparsers.add_parser(AnonMethod.TDG.value, help='tdg anonyization algorithm')

    parser_cb = subparsers.add_parser(AnonMethod.CB.value, help='cb anonyization algorithm')
    parser_cb.add_argument("--cb-alg", choices=['knn', 'kmember', 'oka'],
                           default='knn', help='algorithm for cluster based anonymization')

    parser_arx = [
        subparsers.add_parser(AnonMethod.ARX_SAMPLE.value, help='ARX anonymization (genetic best-effort search)'),
        subparsers.add_parser(AnonMethod.ARX_OPTIMAL.value, help='ARX anonymization (optimal search)'),
        subparsers.add_parser(AnonMethod.ARX_BOTTOM_UP.value, help='ARX anonymization (best-effort bottom-up search)'),
    ]
    for p in parser_arx:
        p.add_argument('--start-s', default=0, type=int, help='initial value for suppression (percent of records)')
        p.add_argument('--stop-s', default=0, type=int, help='last value for suppression (percent of records)')
        p.add_argument('--step-s', default=1, type=int, help='step for increasing suppression')
        p.add_argument('--l-diversity', default=None, type=float, help='enable l-diversity with this l')
        p.add_argument('--l-variant', choices=['distinct', 'entropy'], default='distinct', help='l-diversity variant')
        p.add_argument('--t-closeness', default=None, type=float, help='enable t-closeness with this t (0-1)')
        p.add_argument('--sensitive', default=None,
                       help='sensitive attribute for l-diversity / t-closeness (default: target variable)')

    parser.add_argument('--debug', '-d', action='store_true', help='enable debugging')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='increase output verbosity')

    args = parser.parse_args()

    if args.start_k < 2:
        print("invalid start_k value")
        exit(1)
    if args.stop_k < args.start_k:
        print("stop_k needs to be greater than start_k")
        exit(1)
    if args.step_k < 1 or args.start_k + args.step_k > args.stop_k:
        print("invalid step_k value")
        exit(1)

    main(args)
