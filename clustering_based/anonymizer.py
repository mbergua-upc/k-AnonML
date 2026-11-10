# -*- coding: utf-8 -*-
"""
Entry point used by the experiment script to run cluster-based k-anonymity.

`get_result_one` takes the raw dataset (list of rows), the generalization
hierarchies (att_trees), the value of k, the indexes of the quasi-identifiers
(QI) and sensitive attributes, and the algorithm to use (`type_alg`: 'knn',
'kmember' or 'oka'). It works in three steps:
  1. Moves the QI columns to the front of every row (`reorder_columns`), which is
     the layout expected by `clustering_based_k_anon`.
  2. Runs the clustering algorithm on a deep copy of the data, so the original
     dataset is not modified, and prints k, the NCP (information loss) and the
     running time.
  3. Restores the original column order (`restore_column_order`) and returns the
     anonymized rows, with the QI values generalized.
"""

import copy
import sys
import os

from clustering_based.clustering_based_k_anon import clustering_based_k_anon
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import reorder_columns, restore_column_order

DATA_SELECT = 'a'
TYPE_ALG = 'kmember'
DEFAULT_K = 10
__DEBUG = True


def extend_result(val):
    """
    separated with ',' if it is a list
    """
    if isinstance(val, list):
        return ','.join(val)
    return val


def get_result_one(att_trees, data, k, path, qi_index, SA_index, type_alg):
    "run clustering_based_k_anon for one time, with k=10"
    print("K=%d" % k)
    result, eval_result = clustering_based_k_anon(att_trees, reorder_columns(
        copy.deepcopy(data), qi_index), k, len(qi_index), SA_index, type_alg)
    print("NCP %0.2f" % eval_result[0] + "%")
    print("Running time %0.2f" % eval_result[1] + "seconds")
    return restore_column_order(result, qi_index)
