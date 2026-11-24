"""
MDAV wrapper package: exposes `get_result_mdav`, which anonymizes a dataset with
the MDAV microaggregation algorithm (see anonymizer.py), and `compile_mdav`.
"""
from .anonymizer import compile_mdav, get_result_mdav

__all__ = ['compile_mdav', 'get_result_mdav']
