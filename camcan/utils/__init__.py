"""Various utility tools."""

from .atlas import make_masker_from_atlas
from .shell import run_fs
from .file_parsing import get_area, get_thickness
from .notebook import (run_stacking, run_ridge, plot_pred, plot_error_scatters,
                       plot_learning_curve, plot_barchart, plot_boxplot,
                       plot_error_age, plot_error_segments, run_meg_ridge)

__all__ = ['make_masker_from_atlas', 'run_fs', 'get_area', 'get_thickness',
           'run_stacking', 'run_ridge', 'plot_pred',
           'plot_learning_curve', 'plot_barchart', 'plot_boxplot',
           'plot_error_scatters', 'plot_error_age', 'plot_error_segments',
           'run_meg_ridge']
