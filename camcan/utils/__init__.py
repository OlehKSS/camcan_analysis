"""
The :mod:`camcan.utils` module includes various utilities.
"""

from .atlas import make_masker_from_atlas
from .shell import run_fs
from .file_parsing import get_area, get_thickness
from .stacking import StackingRegressor
from .notebook import (run_stacking, run_ridge, plot_pred,
                       plot_learning_curve, plot_barchart)

__all__ = ['make_masker_from_atlas', 'run_fs', 'get_area', 'get_thickness',
           'StackingRegressor', 'run_stacking','run_ridge', 'plot_pred',
           'plot_learning_curve', 'plot_barchart']

