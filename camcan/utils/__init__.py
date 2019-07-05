"""Various utility tools."""
from .atlas import make_masker_from_atlas
from .evaluation import (run_meg_source_space, run_meg_spoc, run_ridge,
                         run_stacking_source_space, run_stacking_spoc)
from .file_parsing import get_area, get_thickness
from .notebook import (plot_pred, plot_error_scatters,
                       plot_learning_curve, plot_barchart, plot_boxplot,
                       plot_error_age, plot_error_segments)
from .shell import run_fs

__all__ = ['make_masker_from_atlas', 'run_fs', 'get_area', 'get_thickness',
           'plot_pred', 'plot_learning_curve', 'plot_barchart',
           'plot_boxplot', 'plot_error_scatters', 'plot_error_age',
           'plot_error_segments', 'run_ridge', 'run_stacking_spoc',
           'run_meg_spoc', 'run_meg_source_space', 'run_stacking_source_space']
