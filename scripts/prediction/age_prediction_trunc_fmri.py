from itertools import product
import pickle

import pandas as pd
import mne
import numpy as np

from camcan.utils import (run_stacking, run_ridge, plot_pred, plot_learning_curve, plot_boxplot,
                          plot_barchart, plot_error_scatters, plot_error_age, plot_error_segments,
                          run_meg_ridge)


CV = 10
N_JOBS = 4
PANDAS_OUT_FILE = '../../data/age_prediction_exp_data_trunc_fmri.h5'
CONNECTIVITY_KINDS = ('correlation', 'tangent')
TIMESERIES_DURATIONS = (2 * 60, 4 * 60, 6 * 60, 8*60 + 40) # seconds
ATLASES_DESCR = ['modl256', 'basc197']
# store mae, learning curves for summary plots
df_columns = ['MAE', 'STD', 'duration', 'connect', 'atlas'] + [f'MAE{i}' for i in range(0, CV)]
n_rows = len(TIMESERIES_DURATIONS) * len(CONNECTIVITY_KINDS) * len(ATLASES_DESCR)
regression_mae = pd.DataFrame(index=range(0, n_rows), columns=df_columns, dtype=float)
learning_curves = {}

# read information about subjects
subjects_data = pd.read_csv('../../data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age, index=subjects_data.index, dtype=float)

# 643 subjects, each covariance is 9x306x306
# I need to know MEG subjects, as well as structural subjects
meg_data = mne.externals.h5io.read_hdf5('../../data/covs_allch_oas.h5')
meg_subjects = {d['subject'] for d in meg_data if 'subject' in d}
meg_data = None

print(f'Found {len(meg_subjects)} subjects')

# read features

area_data = pd.read_hdf('../../data/structural/structural_data.h5', 
                           key='area')
area_data = area_data.dropna()
# take only subjects that are both in MEG and Structural MRI
structural_subjects = set(area_data.index)
common_subjects = meg_subjects.intersection(structural_subjects)

area_data = None

data_ref = {}
name_map = {'basc197': 'BASC 197', 'modl256': 'MODL 256',
            'correlation': 'r2z', 'tangent': 'tan'}

for i, (d, c, a) in enumerate(product(TIMESERIES_DURATIONS, CONNECTIVITY_KINDS, ATLASES_DESCR)):
    hdf5_key = f'ts_{d}_{c}_{a}'
    connect_data = pd.read_hdf(f'../../data/connectivity/connect_data_truncated.h5',
                                key=hdf5_key)
    connect_data = connect_data.loc[common_subjects]
    df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores \
        = run_ridge(connect_data, subjects_data, cv=CV, n_jobs=N_JOBS)

    arr_mae = -arr_mae
    mae = arr_mae.mean()
    std = arr_mae.std()
    print('%s MAE: %.2f, STD %.2f' % (hdf5_key, mae, std))

    regression_mae.iloc[i] = [mae, std, d, c, a] + arr_mae.tolist()
    key = f'Connectivity Matrix, {name_map[a]} {name_map[c]}, {d}'
    subjects_predictions.loc[df_pred.index, key] = df_pred[0]
    learning_curves[key] = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores
    }

# save results
with open('../../data/learning_curves_trunc_fmri.pkl', 'wb') as handle:
    pickle.dump(learning_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

subjects_predictions.to_hdf(PANDAS_OUT_FILE, key='predictions', complevel=9)
regression_mae.to_hdf(PANDAS_OUT_FILE, key='regression', complevel=9)
