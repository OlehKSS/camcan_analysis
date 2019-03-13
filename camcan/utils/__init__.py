"""
The :mod:`camcan.utils` module includes various utilities.
"""

from .atlas import make_masker_from_atlas
from .shell import run_fs
from .file_parsing import get_area, get_thickness

__all__ = ['make_masker_from_atlas', 'run_fs', 'get_area', 'get_thickness']
