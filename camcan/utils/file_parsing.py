"""Utility functions for parcinging Freesurfer output files"""
import nibabel as nb
import numpy as np


def _vectorize_fs_surf(file_path):
    """
    Read surface information from a file and turn it into a vector.
    
    Parameters
    ----------
    file_path : str
        The path to a file with surface data.
    
    Returns
    -------
    vectorized_data : numpy.ndarray
        Extracted data.
    """
    img = nb.load(file_path)
    in_data = img.get_data().squeeze()

    return in_data


def get_area(subject_dir):
    """
    Read area information for the given subject and turn it into a vector.
    Data for left and right hemispheres are concatenated.
    
    Parameters
    ----------
    subject_dir : str
        The directory to files with surface data.
    
    Returns
    -------
    : numpy.ndarray
        Extracted data.
    """
    AREA_FILES = ('lh.area.mgh',  'rh.area.mgh')
    
    lh_data = _vectorize_fs_surf(join(subject_dir, AREA_FILES[0]))
    rh_data = _vectorize_fs_surf(join(subject_dir, AREA_FILES[1]))
    
    return np.concatenate((lh_data, rh_data), 0)


def get_thickness(subject_dir):
    """
    Read thickness information for the given subject and turn it into a vector.
    Data for left and right hemispheres are concatenated.
    
    Parameters
    ----------
    subject_dir : str
        The directory to files with surface data.
    
    Returns
    -------
    : numpy.ndarray
        Extracted data.
    """
    THICKNESS_FILES = ('rh.thickness.mgh', 'lh.thickness.mgh')
    
    lh_data = _vectorize_fs_surf(join(subject_dir, THICKNESS_FILES[0]))
    rh_data = _vectorize_fs_surf(join(subject_dir, THICKNESS_FILES[1]))
    
    return np.concatenate((lh_data, rh_data), 0)
    