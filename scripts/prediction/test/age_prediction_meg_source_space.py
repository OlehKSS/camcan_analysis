"""Age prediction using MRI, fMRI and MEG data."""
from itertools import permutations
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from camcan.utils import (run_stacking_source_space, run_ridge,
                          run_meg_source_space)
from threadpoolctl import threadpool_limits


# common subjects 574
CV = 10
N_JOBS = 4
PANDAS_OUT_FILE = '../../data/age_prediction_exp_data.h5'
STRUCTURAL_DATA = '../../data/structural/structural_data.h5'
CONNECT_DATA_CORR = '../../data/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = '../../data/connectivity/connect_data_tangent.h5'
MEG_SOURCE_SPACE_DATA = '../../data/meg_source_space_data.h5'
FREQ_BANDS = ('alpha',
              'beta_high',
              'beta_low',
              'delta',
              'gamma_high',
              'gamma_lo',
              'gamma_mid',
              'low',
              'theta')
# store mae, learning curves for summary plots
regression_mae = pd.DataFrame(columns=range(0, CV), dtype=float)
regression_r2 = pd.DataFrame(columns=range(0, CV), dtype=float)
learning_curves = {}

# read information about subjects
subjects_data = pd.read_csv('../../data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age,
                                    index=subjects_data.index,
                                    dtype=float)

# 595 subjects
meg_data = pd.read_hdf(MEG_SOURCE_SPACE_DATA, key='meg')
meg_subjects = set(meg_data['subject'])

# df_pred, mae, r2, train_sizes, train_scores, test_scores =\
#     run_meg_source_space(meg_data, subjects_data, cv=CV, fbands=FREQ_BANDS)
# read features
area_data = pd.read_hdf(STRUCTURAL_DATA, key='area')
thickness_data = pd.read_hdf(STRUCTURAL_DATA, key='thickness')
volume_data = pd.read_hdf(STRUCTURAL_DATA, key='volume')

area_data = area_data.dropna()
thickness_data = thickness_data.dropna()
volume_data = volume_data.dropna()

# take only subjects that are both in MEG and Structural MRI
structural_subjects = set(area_data.index)
common_subjects = meg_subjects.intersection(structural_subjects)

area_data = area_data.loc[common_subjects]
thickness_data = thickness_data.loc[common_subjects]
volume_data = volume_data.loc[common_subjects]
meg_data = meg_data[meg_data.subject.isin(common_subjects)]

# read connectivity data
connect_data_tangent_basc = pd.read_hdf(CONNECT_DATA_TAN, key='basc197')
connect_data_r2z_basc = pd.read_hdf(CONNECT_DATA_CORR, key='basc197')
connect_data_tangent_modl = pd.read_hdf(CONNECT_DATA_TAN, key='modl256')
connect_data_r2z_modl = pd.read_hdf(CONNECT_DATA_CORR, key='modl256')

# use only common subjects
connect_data_tangent_basc = connect_data_tangent_basc.loc[common_subjects]
connect_data_r2z_basc = connect_data_r2z_basc.loc[common_subjects]
connect_data_tangent_modl = connect_data_tangent_modl.loc[common_subjects]
connect_data_r2z_modl = connect_data_r2z_modl.loc[common_subjects]

print('Data was read successfully.')

# test on one modality, subject suffling
# test modality shuffling
cv = KFold(n_splits=CV, shuffle=True, random_state=42)

n_iter = 2

# test subcortical volumes regression
test_sv = np.zeros(n_iter)

for i in range(n_iter):
    data = volume_data.sample(frac=1)
    df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
        run_ridge(data, subjects_data, cv=cv, n_jobs=N_JOBS)
    arr_mae = -arr_mae
    test_sv[i] = arr_mae.mean()

print(f'Test subcortical volumes {test_sv}')
# test meg regression
# test_meg = np.zeros(n_iter)

# for i in range(n_iter):
#     data = meg_data.sample(frac=1)
#     df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
#     run_meg_source_space(data, subjects_data, cv=cv,
#                             fbands=FREQ_BANDS, n_jobs=N_JOBS)
#     arr_mae = -arr_mae
#     test_meg[i] = arr_mae.mean()

# test stacking
test_stacking_data = [('volume', volume_data),
                      ('basc', connect_data_tangent_basc),
                      ('meg', meg_data)]
perm_data = list(permutations(test_stacking_data))
test_stacking = np.zeros(len(perm_data))

with threadpool_limits(limits=N_JOBS, user_api='blas'):
    for i, data in enumerate(perm_data):
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
            run_stacking_source_space(data, subjects_data, cv=cv,
                                    fbands=FREQ_BANDS, n_jobs=N_JOBS)
        arr_mae = -arr_mae
        test_stacking[i] = arr_mae.mean()

print(f'Test stacking {test_stacking}')
