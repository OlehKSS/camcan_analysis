# Line 88 what is subjects2 for? why do we take the reduced number of sub
"""Age prediction using MRI, fMRI and MEG data."""
import pickle

import pandas as pd

from camcan.utils import (run_stacking_source_space, run_ridge,
                          run_meg_source_space)


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

data_ref = {
    'Cortical Surface Area': area_data,
    'Cortical Thickness': thickness_data,
    'Subcortical Volumes': volume_data,
    'Connectivity Matrix, BASC 197 tan': connect_data_tangent_basc,
    'Connectivity Matrix, BASC 197 r2z': connect_data_r2z_basc,
    'Connectivity Matrix, MODL 256 tan': connect_data_tangent_modl,
    'Connectivity Matrix, MODL 256 r2z': connect_data_r2z_modl,
    'MEG': meg_data,
    'MEG, Cortical Surface Area Stacked-multimodal': [('area', area_data),
                                                      ('meg', meg_data)],
    'MEG, Cortical Thickness Stacked-multimodal': [('thickness',
                                                    thickness_data),
                                                   ('meg', meg_data)],
    'MEG, Subcortical Volumes Stacked-multimodal': [('volume', volume_data),
                                                    ('meg', meg_data)],
    'MEG, BASC 197 tan Stacked-multimodal': [('basc',
                                              connect_data_tangent_basc),
                                             ('meg', meg_data)],
    'MEG, MODL 256 r2z Stacked-multimodal': [('modl', connect_data_r2z_modl),
                                             ('meg', meg_data)],
    'MRI Stacked': [('area', area_data), ('thickness', thickness_data),
                    ('volume', volume_data)],
    'fMRI Stacked': [('basc', connect_data_tangent_basc),
                     ('modl', connect_data_r2z_modl)],
    'MRI, fMRI Stacked-multimodal': [('area', area_data),
                                     ('thickness', thickness_data),
                                     ('volume', volume_data),
                                     ('basc',
                                      connect_data_tangent_basc),
                                     ('modl', connect_data_r2z_modl)],
    'MEG, MRI Stacked-multimodal': [('area', area_data),
                                    ('thickness', thickness_data),
                                    ('volume', volume_data),
                                    ('meg', meg_data)],
    'MEG, fMRI Stacked-multimodal': [('basc', connect_data_tangent_basc),
                                     ('modl', connect_data_r2z_modl),
                                     ('meg', meg_data)],
    'MEG, MRI, fMRI Stacked-multimodal': [('area', area_data),
                                          ('thickness', thickness_data),
                                          ('volume', volume_data),
                                          ('basc', connect_data_tangent_basc),
                                          ('modl', connect_data_r2z_modl),
                                          ('meg', meg_data)]
}

for key, data in data_ref.items():
    if 'Stack' in key:
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
            run_stacking_source_space(data, subjects_data, cv=CV,
                                      fbands=FREQ_BANDS)
    elif key == 'MEG':
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
            run_meg_source_space(data, subjects_data, cv=CV, fbands=FREQ_BANDS)
    else:
        df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
            run_ridge(data, subjects_data, cv=CV, n_jobs=N_JOBS)

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
