# -*- coding: utf-8 -*-

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
