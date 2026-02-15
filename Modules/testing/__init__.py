"""
Testing utilities for TD-SCHA.

This module provides test data and utilities for documentation examples
and doctests.
"""

from .test_data import (
    get_test_data_path,
    load_test_ensemble,
    create_test_lanczos,
    get_test_mode_frequencies,
    HAS_DEPENDENCIES,
)

__all__ = [
    'get_test_data_path',
    'load_test_ensemble',
    'create_test_lanczos',
    'get_test_mode_frequencies',
    'HAS_DEPENDENCIES',
]