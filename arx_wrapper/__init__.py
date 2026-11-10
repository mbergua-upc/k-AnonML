"""
ARX wrapper package: exposes `get_result_arx`, which anonymizes a dataset with the
ARX tool (see anonymizer.py).
"""
from .anonymizer import get_result_arx

__all__ = ['get_result_arx']