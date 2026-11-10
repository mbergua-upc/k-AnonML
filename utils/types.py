# -*- coding: utf-8 -*-

"""
Data Types and Result Structures

This module defines the main enumerations and data structures used throughout
the anonymization and machine learning evaluation framework.

CLASSES:
    - Dataset (Enum): Defines the supported datasets (CMC, MGM, Adult and
      CAHousing) and provides string representations of their identifiers.

    - AnonMethod (Enum): Defines the supported anonymization methods
      (OLA, Mondrian, TDG, Clustering-Based and the three ARX search strategies).

    - Classifier (Enum): Defines the supported machine learning classifiers
      (Random Forest, KNN, SVM and XGBoost).

    - MLRes (namedtuple): Stores the machine learning evaluation results
      (accuracy, precision, recall and F1-score), initializing each metric
      with an independent list to collect experimental results.
"""

from collections import namedtuple
from enum import Enum


class Dataset(Enum):
    CMC = 'cmc'
    MGM = 'mgm'
    ADULT = 'adult'
    CAHOUSING = 'cahousing'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class AnonMethod(Enum):
    OLA = 'ola'
    MONDRIAN = 'mondrian'
    TDG = 'tdg'
    CB = 'cb'
    ARX_SAMPLE = 'arx_sample'
    ARX_OPTIMAL = 'arx_optimal'
    ARX_BOTTOM_UP = 'arx_bottom_up'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class Classifier(Enum):
    RF = 'rf'
    KNN = 'knn'
    SVM = 'svm'
    XGB = 'xgb'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class MLRes(namedtuple("MLRes", ['acc', 'precision', 'recall', 'f1_score'])):
    __slots__ = ()

    def __new__(cls, acc=None, precision=None, recall=None, f1_score=None):
        # fresh lists on every call: shared mutable defaults leaked results between runs (e.g. between s values)
        return super().__new__(
            cls,
            ['accuracy'] if acc is None else acc,
            ['precision'] if precision is None else precision,
            ['recall'] if recall is None else recall,
            ['f1'] if f1_score is None else f1_score,
        )
