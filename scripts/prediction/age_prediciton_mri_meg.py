import pickle

import pandas as pd
import mne
import numpy as np

from camcan.utils import (run_stacking, run_ridge, plot_pred, plot_learning_curve, plot_boxplot,
                          plot_barchart, plot_error_scatters, plot_error_age, plot_error_segments,
                          run_meg_ridge)


CV = 10
N_JOBS = 4
PANDAS_OUT_FILE = '../../data/age_prediction_exp_data.h5'
# store mae, learning curves for summary plots
regression_mae = pd.DataFrame(columns=range(0, CV), dtype=float)
regression_r2 = pd.DataFrame(columns=range(0, CV), dtype=float)
learning_curves = {}

# read information about subjects
subjects_data = pd.read_csv('../../data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age, index=subjects_data.index, dtype=float)

# 643 subjects, each covariance is 9x306x306
meg_data = mne.externals.h5io.read_hdf5('../../data/covs_allch_oas.h5')
meg_subjects = {d['subject'] for d in meg_data if 'subject' in d}

print(f'Found {len(meg_data)} subjects')
print(f'A covarince matrix shape is {meg_data[0]["covs"].shape}')

# read features

area_data = pd.read_hdf('../../data/structural/structural_data.h5', 
                           key='area')
thickness_data = pd.read_hdf('../../data/structural/structural_data.h5',
                               key='thickness')
volume_data = pd.read_hdf('../../data/structural/structural_data.h5',
                            key='volume')

area_data = area_data.dropna()
thickness_data = thickness_data.dropna()
volume_data = volume_data.dropna()

# take only subjects that are both in MEG and Structural MRI
structural_subjects = set(area_data.index)
common_subjects = meg_subjects.intersection(structural_subjects)

area_data = area_data.loc[common_subjects]
thickness_data = thickness_data.loc[common_subjects]
volume_data = volume_data.loc[common_subjects]

# read connectivity data
connect_data_tangent_basc = pd.read_hdf('../../data/connectivity/connect_data_tangent.h5',
                              key='basc197')
connect_data_r2z_basc = pd.read_hdf('../../data/connectivity/connect_data_correlation.h5',
                              key='basc197')
connect_data_tangent_modl = pd.read_hdf('../../data/connectivity/connect_data_tangent.h5',
                              key='modl256')
connect_data_r2z_modl = pd.read_hdf('../../data/connectivity/connect_data_correlation.h5',
                              key='modl256')
# use only common subjects
connect_data_tangent_basc = connect_data_tangent_basc.loc[common_subjects]
connect_data_r2z_basc = connect_data_r2z_basc.loc[common_subjects]
connect_data_tangent_modl = connect_data_tangent_modl.loc[common_subjects]
connect_data_r2z_modl = connect_data_r2z_modl.loc[common_subjects]

FREQ_BANDS = [(0.1, 1.5),  # low
              (1.5, 4.0),  # delta
              (4.0, 8.0),  # theta
              (8.0, 15.0),  # alpha
              (15.0, 26.0),  # beta_low
              (26.0, 35.0),  # beta_high
              (35.0, 50.0),  # gamma_low
              (50.0, 74.0),  # gamma_mid
              (76.0, 120.0)]  # gamma_high

data_ref = {
    'Cortical Surface Area': area_data,
    'Cortical Thickness': thickness_data,
    'Subcortical Volumes': volume_data,
    'Connectivity Matrix, BASC 197 tan': connect_data_tangent_basc,
    'Connectivity Matrix, BASC 197 r2z': connect_data_r2z_basc,
    'Connectivity Matrix, MODL 256 tan': connect_data_tangent_modl,
    'Connectivity Matrix, MODL 256 r2z': connect_data_r2z_modl,
    'MEG': meg_data,
    'MEG, Cortical Surface Area Stacked-multimodal': [('area', area_data), ('meg', meg_data)],
    'MEG, Cortical Thickness Stacked-multimodal': [('thickness', thickness_data), ('meg', meg_data)],
    'MEG, Subcortical Volumes Stacked-multimodal': [('volume', volume_data), ('meg', meg_data)],
    'MEG, BASC 197 tan Stacked-multimodal': [('basc', connect_data_tangent_basc), ('meg', meg_data)],
    'MEG, MODL 256 r2z Stacked-multimodal': [('modl', connect_data_r2z_modl), ('meg', meg_data)],
    'MRI Stacked': [('area', area_data), ('thickness', thickness_data), ('volume', volume_data)],
    'fMRI Stacked': [('basc', connect_data_tangent_basc), ('modl', connect_data_r2z_modl)],
    'MRI, fMRI Stacked-multimodal': [('area', area_data), ('thickness', thickness_data), ('volume', volume_data),
                                     ('basc', connect_data_tangent_basc), ('modl', connect_data_r2z_modl)],
    'MEG, MRI Stacked-multimodal': [('area', area_data), ('thickness', thickness_data), ('volume', volume_data),
                                    ('meg', meg_data)],
    'MEG, fMRI Stacked-multimodal': [('basc', connect_data_tangent_basc), ('modl', connect_data_r2z_modl),
                                     ('meg', meg_data)],
    'MEG, MRI, fMRI Stacked-multimodal': [('area', area_data), ('thickness', thickness_data), ('volume', volume_data),
                                          ('basc', connect_data_tangent_basc), ('modl', connect_data_r2z_modl),
                                          ('meg', meg_data)]
}

for key, data in data_ref.items():
    if 'Stack' in key:
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores \
        = run_stacking(data, subjects_data, cv=CV, fbands=FREQ_BANDS)
    elif key == 'MEG':
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores = \
        run_meg_ridge(data, subjects_data, cv=CV, fbands=FREQ_BANDS)
    else:
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores \
        = run_ridge(data, subjects_data, cv=CV, n_jobs=N_JOBS)

    arr_mae = -arr_mae
    mae = arr_mae.mean()
    std = arr_mae.std()
    print('%s MAE: %.2f, STD %.2f' % (key, mae, std))

    regression_mae.loc[key] = arr_mae
    regression_r2.loc[key] = arr_r2
    subjects_predictions.loc[df_pred.index, key] = df_pred[0]
    learning_curves[key] = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores
    }

# save results
with open('../../data/learning_curves.pkl', 'wb') as handle:
    pickle.dump(learning_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

subjects_predictions.to_hdf(PANDAS_OUT_FILE, key='predictions', complevel=9)
regression_mae.to_hdf(PANDAS_OUT_FILE, key='regression', complevel=9)
regression_r2.to_hdf(PANDAS_OUT_FILE, key='r2', complevel=9)
