"""Script for extracting structural data from the data processed with FreeSurfer"""
import os
from os.path import isdir, join

import joblib
from joblib import Parallel, delayed

from camcan.preprocessing import get_structural_data

# import pdb; pdb.set_trace()

# test functions on MRI data
CAMCAN_CONNECTIVITY = '/storage/data/camcan/camcan_connectivity'
CAMCAN_FREESURFER = '/storage/store/data/camcan-mne/freesurfer'
OUT_DIR = '/storage/tompouce/okozynet/camcan/structural'
N_JOBS = 5
# list of subjects that we have connectivity data for
subjects = [d[4:] for d in os.listdir(CAMCAN_CONNECTIVITY) if isdir(join(CAMCAN_CONNECTIVITY, d))]
subjects = ['CC221585']
structural_data = Parallel(n_jobs=N_JOBS, verbose=1)(
                           delayed(get_structural_data)(CAMCAN_FREESURFER, s, OUT_DIR)
                           for s in subjects)
