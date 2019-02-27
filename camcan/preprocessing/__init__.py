"""
The :mod:`camcan.preprocessing` module includes methods to extract time series.
"""
from .temporal_series import extract_timeseries
from .connectivity import extract_connectivity
from .structural import get_structural_data

__all__ = ['extract_timeseries',
           'extract_connectivity',
           'get_structural_data']
