"""
The :mod:`camcan.utils` module includes various utilities.
"""

from .atlas import make_masker_from_atlas
from .shell import run_fs
from .file_parsing import get_area, get_thickness
from .stacking import StackingRegressor
from .notebook import (run_stacking, run_ridge, plot_pred, plot_error_scatters,
                       plot_learning_curve, plot_barchart, plot_boxplot, plot_error_age,
                       plot_error_segments)

__all__ = ['make_masker_from_atlas', 'run_fs', 'get_area', 'get_thickness',
           'StackingRegressor', 'run_stacking','run_ridge', 'plot_pred',
           'plot_learning_curve', 'plot_barchart', 'plot_boxplot',
           'plot_error_scatters', 'plot_error_age', 'plot_error_segments']

