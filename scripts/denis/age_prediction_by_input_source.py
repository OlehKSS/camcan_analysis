"""Age prediction using MRI, fMRI and MEG data."""
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mne.externals import h5io

from camcan.utils import (run_stacking, run_ridge)
from threadpoolctl import threadpool_limits


# common subjects 574
CV = 10
N_JOBS = 2
PANDAS_OUT_FILE = './data/age_prediction_exp_data_denis.h5'
STRUCTURAL_DATA = './data/structural/structural_data.h5'
CONNECT_DATA_CORR = './data/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = './data/connectivity/connect_data_tangent.h5'
MEG_SOURCE_SPACE_DATA = './data/meg_source_space_data.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_ENV_CORR = './data/all_power_envelopes.h5'

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
subjects_data = pd.read_csv('./data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age,
                                    index=subjects_data.index,
                                    dtype=float)

# 595 subjects
meg_data = pd.read_hdf(MEG_SOURCE_SPACE_DATA, key='meg')

columns_to_exclude = ('band', 'fmax', 'fmin', 'subject')
parcellation_labels = [c for c in meg_data.columns if c
                       not in columns_to_exclude]
band_data = [meg_data[meg_data.band == bb].set_index('subject')[
                parcellation_labels] for bb in FREQ_BANDS]
meg_data = pd.concat(band_data, axis=1, join='inner', sort=False)
meg_subjects = set(meg_data.index)

meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')
meg_extra = meg_extra.reset_index()

meg_peaks = pd.read_csv(MEG_PEAKS)

meg_envelopes = h5io.read_hdf5(MEG_ENV_CORR)

C_index = np.eye(448, dtype=np.bool)
C_index = np.invert(C_index[np.triu_indices(448)])

meg_envelope_subjects = list(meg_envelopes)

meg_envelope_alpha_cov = pd.DataFrame(
    [meg_envelopes[sub]['alpha'].pop('cov') for sub in
     meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta1_cov = pd.DataFrame(
    [meg_envelopes[sub]['beta_low'].pop('cov') for sub in
     meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta2_cov = pd.DataFrame(
    [meg_envelopes[sub]['beta_high'].pop('cov')
     for sub in meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_alpha_corr = pd.DataFrame(
    [meg_envelopes[sub]['alpha'].pop('corr')[C_index] for sub in
     meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta1_corr = pd.DataFrame(
    [meg_envelopes[sub]['beta_low'].pop('corr')[C_index] for sub in
     meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta2_corr = pd.DataFrame(
    [meg_envelopes[sub]['beta_high'].pop('corr')[C_index]
     for sub in meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_alpha_orth = pd.DataFrame(
    [meg_envelopes[sub]['alpha'].pop('corr_orth')[C_index]
     for sub in meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta1_orth = pd.DataFrame(
    [meg_envelopes[sub]['beta_low'].pop('corr_orth')[C_index]
     for sub in meg_envelope_subjects],
    index=meg_envelope_subjects)

meg_envelope_beta2_orth = pd.DataFrame(
    [meg_envelopes[sub]['beta_high'].pop('corr_orth')[C_index]
     for sub in meg_envelope_subjects],
    index=meg_envelope_subjects)


meg_subjects = (meg_subjects.intersection(meg_extra['subject'])
                            .intersection(meg_peaks['subject']))

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
meg_extra = meg_extra[meg_extra.subject.isin(common_subjects)]
meg_peaks = meg_peaks[meg_peaks.subject.isin(common_subjects)]

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
    # 'Connectivity Matrix, BASC 197 tan': connect_data_tangent_basc,
    # 'Connectivity Matrix, BASC 197 r2z': connect_data_r2z_basc,
    'Connectivity Matrix, MODL 256 tan': connect_data_tangent_modl,
    # 'Connectivity Matrix, MODL 256 r2z': connect_data_r2z_modl,
    'MEG': meg_data,
    'MEG alpha cov': meg_envelope_alpha_cov,
    'MEG alpha corr': meg_envelope_alpha_corr,
    'MEG alpha orth': meg_envelope_alpha_orth,
    'MEG beta1 cov': meg_envelope_beta1_cov,
    'MEG beta1 corr': meg_envelope_beta1_corr,
    'MEG beta1 orth': meg_envelope_beta1_orth,
    'MEG beta2 cov': meg_envelope_beta2_cov,
    'MEG beta2 corr': meg_envelope_beta2_corr,
    'MEG beta2 orth': meg_envelope_beta2_orth,
    'MEG 1/f low': meg_extra.set_index("subject")[
        [cc for cc in meg_extra.columns if '1f_low' in cc]],
    'MEG 1/f gamma': meg_extra.set_index("subject")[
        [cc for cc in meg_extra.columns if '1f_gamma' in cc]],
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
                                      connect_data_tangent_basc)],
    'MEG, MRI Stacked-multimodal': [('area', area_data),
                                    ('thickness', thickness_data),
                                    ('volume', volume_data),
                                    ('meg', meg_data)],
    'MEG, fMRI Stacked-multimodal': [('basc', connect_data_tangent_basc),
                                     ('meg', meg_data)],
    'MEG, MRI, fMRI Stacked-multimodal': [('area', area_data),
                                          ('thickness', thickness_data),
                                          ('volume', volume_data),
                                          ('basc', connect_data_tangent_basc),
                                          ('meg', meg_data)]
}

cv = KFold(n_splits=CV, shuffle=True, random_state=42)


def run_ridge_boost(data, subjects_data, cv=10, alphas=None, train_sizes=None,
                    n_jobs=None):
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import (cross_val_score,
                                         cross_val_predict,
                                         learning_curve,
                                         ShuffleSplit, check_cv)
    # prepare data, subjects age
    subjects = data.index.values
    y = subjects_data.loc[subjects].age.values
    X = data.values

    reg = make_pipeline(StandardScaler(), RidgeCV(alphas))

    cv = check_cv(cv)
    # mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
    #                       cv=cv, n_jobs=n_jobs)
    # r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv, n_jobs=n_jobs)
    # y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)
    # fold = _get_fold_indices(cv, X, y)
    for train, test in cv.s


    df_pred = pd.DataFrame(dict(y=y_pred, fold=fold), index=subjects,
                           dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores

with threadpool_limits(limits=N_JOBS, user_api='blas'):
    for key, data in data_ref.items():
        if 'Stack' in key:
            continue
            if False:
                (df_pred, arr_mae, arr_r2, train_sizes, train_scores,
                 test_scores) = run_stacking(
                    data, subjects_data, cv=cv, n_jobs=N_JOBS)
        else:
            df_pred, arr_mae, arr_r2, train_sizes, train_scores, test_scores =\
                run_ridge(data, subjects_data, cv=cv, n_jobs=N_JOBS)

        arr_mae = -arr_mae
        mae = arr_mae.mean()
        std = arr_mae.std()
        print('%s MAE: %.2f, STD %.2f' % (key, mae, std))

        regression_mae.loc[key] = arr_mae
        regression_r2.loc[key] = arr_r2
        subjects_predictions.loc[df_pred.index, key] = df_pred['y']
        subjects_predictions.loc[df_pred.index, 'fold_idx'] = df_pred['fold']
        learning_curves[key] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        }

# save results
with open('./data/learning_curves_denis.pkl', 'wb') as handle:
    pickle.dump(learning_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

subjects_predictions.to_hdf(PANDAS_OUT_FILE, key='predictions', complevel=9)
regression_mae.to_hdf(PANDAS_OUT_FILE, key='regression', complevel=9)
regression_r2.to_hdf(PANDAS_OUT_FILE, key='r2', complevel=9)
