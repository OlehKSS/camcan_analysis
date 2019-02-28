"""Script for extracting structural data from the data processed with FreeSurfer"""
import os
from os.path import isdir

import joblib
from joblib import Parallel, delayed

from camcan.preprocessing import get_structural_data

# test functions on MRI data
CAMCAN_CONNECTIVITY = '/home/mehdi/data/camcan/camcan_connectivity'
CAMCAN_FREESURFER = '/storage/store/data/camcan-mne/freesurfer'
OUT_DIR = '/storage/tompouce/okozynet/camcan/structural'
N_JOBS = 5
# list of subjects that we have connectivity data for
subjects = [d for d in os.listdir(CAMCAN_CONNECTIVITY) if isdir(d) and 'CC' in d] 

structural_data = Parallel(n_jobs=N_JOBS, verbose=1)(
                           delayed(get_structural_data)(CAMCAN_FREESURFER, s, OUT_DIR)
                           for s in subjects)
