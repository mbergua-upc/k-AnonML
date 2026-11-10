# -*- coding: utf-8 -*-
"""
Entry point used by the experiment script to run Top-Down Greedy k-anonymity.

`tdg_get_result_one` takes the raw dataset (list of rows), the generalization
hierarchies (att_trees), the value of k and the indexes of the quasi-identifiers
(QI) and sensitive attributes. It works in three steps:
  1. Moves the QI columns to the front of every row (`reorder_columns`), which is
     the layout expected by `Top_Down_Greedy_Anonymization`.
  2. Runs the algorithm on a deep copy of the data, so the original dataset is
     not modified, and prints k, the NCP (information loss) and the running time.
  3. Restores the original column order (`restore_column_order`) and returns the
     anonymized rows, with the QI values generalized.
"""

import copy
import sys
import os

from top_down_greedy.top_down_greedy_anonymization import \
    Top_Down_Greedy_Anonymization
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import reorder_columns, restore_column_order

DATA_SELECT = 'a'


def tdg_get_result_one(att_trees, data, k, path, qi_index, SA_index):
    "run Top_Down_Greedy_Anonymization for one time, with k=10"
    print("K=%d" % k)
    result, eval_result = Top_Down_Greedy_Anonymization(att_trees, reorder_columns(
        copy.deepcopy(data), qi_index), k, len(qi_index), SA_index)
    print("NCP %0.2f" % eval_result[0] + "%")
    print("Running time %0.2f" % eval_result[1] + "seconds")
    return restore_column_order(result, qi_index)
