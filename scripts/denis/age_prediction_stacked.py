"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, LeaveOneGroupOut)
import xgboost as xgb

FIG_OUT_PATH = './data/figures/'
PREDICTIONS = './data/age_prediction_exp_data_denis.h5'
SCORES = './data/age_prediction_scores_denis.csv'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'

data = pd.read_hdf(PREDICTIONS, key='predictions')
data = data.dropna()

# Add extra dfeatures

subjects = set(data.index)
meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')
meg_extra = meg_extra.reset_index()
meg_peaks = pd.read_csv(MEG_PEAKS)
meg_peaks2 = pd.read_csv(MEG_PEAKS2)

meg_subjects = (subjects.intersection(meg_extra['subject'])
                        .intersection(meg_peaks['subject'])
                        .intersection(meg_peaks2['subject']))

meg_extra = meg_extra.set_index('subject')
meg_peaks = meg_peaks.set_index('subject')
meg_peaks2 = meg_peaks2.set_index('subject')

data = data.loc[list(meg_subjects)]
meg_extra = meg_extra.loc[list(meg_subjects)]
meg_peaks = meg_peaks.loc[list(meg_subjects)]
meg_peaks2 = meg_peaks2.loc[list(meg_subjects)]

assert np.all(meg_extra.index.values == data.index.values)
assert np.all(meg_peaks.index.values == data.index.values)
assert np.all(meg_peaks.index.values == meg_extra.index.values)
assert np.all(meg_peaks2.index.values == meg_extra.index.values)


data['MEG alpha peak'] = meg_extra['alpha_peak'].values
data['MEG latency aud'] = meg_peaks['aud'].values
data['MEG latency vis'] = meg_peaks['vis'].values
data['MEG latency audvis'] = meg_peaks2['audvis'].values

stacked_keys = {
    'MEG all': [
        'MEG',
        'MEG 1/f low',
        'MEG 1/f gamma',
        'MEG alpha peak',
        'MEG latency aud',
        'MEG latency vis',
        'MEG latency audvis',
        'MEG alpha corr',
        'MEG alpha orth',
        'MEG beta1 corr',
        'MEG beta1 orth',
        'MEG beta2 corr',
        'MEG beta2 orth'
    ],
    # 'full': [
    #     'MEG',
    #     'MEG 1/f low',
    #     'MEG 1/f gamma',
    #     'MEG alpha peak',
    #     'MEG latency aud',
    #     'MEG latency vis',
    #     'Cortical Surface Area',
    #     'Cortical Thickness',
    #     'Subcortical Volumes',
    #     'Connectivity Matrix, MODL 256 tan',
    #     'MEG alpha corr',
    #     'MEG alpha orth',
    #     'MEG beta1 corr',
    #     'MEG beta1 orth',
    #     'MEG beta2 corr',
    #     'MEG beta2 orth'
    # ],

    # 'MRI': [
    #     'Cortical Surface Area',
    #     'Cortical Thickness',
    #     'Subcortical Volumes',
    #     'Connectivity Matrix, MODL 256 tan',
    # ],

    # 'MEG, Cortical Surface Area Stacked-multimodal': ['Cortical Surface Area',
    #                                                   'MEG'],
    # 'MEG, Cortical Thickness Stacked-multimodal': ['Cortical Thickness', 'MEG'],
    # 'MEG, Subcortical Volumes Stacked-multimodal': ['Subcortical Volumes', 'MEG'],
    # 'MEG, BASC 197 tan Stacked-multimodal': ['Connectivity Matrix, BASC 197 tan',
    #                                          'MEG'],
    # 'MEG, MODL 256 r2z Stacked-multimodal': ['Connectivity Matrix, MODL 256 r2z',
    #                                          'MEG'],
    # 'MRI Stacked': ['Cortical Surface Area', 'Cortical Thickness',
    #                 'Subcortical Volumes'],
    # 'fMRI Stacked': ['Connectivity Matrix, BASC 197 tan',
    #                  'Connectivity Matrix, MODL 256 r2z'],
    # 'MRI, fMRI Stacked-multimodal': ['Cortical Surface Area',
    #                                  'Cortical Thickness',
    #                                  'Subcortical Volumes',
    #                                  'Connectivity Matrix, BASC 197 tan'],
    # 'MEG, MRI Stacked-multimodal': ['Cortical Surface Area',
    #                                 'Cortical Thickness',
    #                                 'Subcortical Volumes',
    #                                 'MEG'],
    # 'MEG, fMRI Stacked-multimodal': ['Connectivity Matrix, BASC 197 tan',
    #                                  'MEG'],
    # 'MEG, MRI, fMRI Stacked-multimodal': ['Cortical Surface Area',
    #                                       'Cortical Thickness',
    #                                       'Subcortical Volumes',
    #                                       'Connectivity Matrix, BASC 197 tan',
    #                                       'MEG']
}

key_labels = {'Cortical Surface Area': 'Cortical Surface Area',
              'Cortical Thickness': 'Cortical Thickness',
              'Subcortical Volumes': 'Subcortical Volumes',
              # 'Connectivity Matrix, BASC 197 tan': 'BASC 197 tan',
              'Connectivity Matrix, MODL 256 tan': 'MODL 256 tan',
              # 'Connectivity Matrix, MODL 256 r2z': 'MODL 256 r2z',
              # 'Connectivity Matrix, BASC 197 r2z': 'BASC 197 r2z',
              'MEG': 'MEG',
              }


def _get_mae(predictions, key):
    scores = []
    for fold_idx, df in predictions.groupby('fold_idx'):
        scores.append(np.mean(np.abs(df[key] - df['age'])))
    return scores


regression_scores = pd.DataFrame()

for key in key_labels:
    regression_scores[key] = _get_mae(data, key)

# Do the stacking:

y = data['age']
fold_idx = data['fold_idx']
# data = data.drop(['fold_idx', 'age'], axis=1)


for key, val in stacked_keys.items():
    X = data[val].values

    reg = RandomForestRegressor(n_estimators=500, max_depth=5,
                                random_state=42)
    # reg = GradientBoostingRegressor(
    #     n_estimators=500, learning_rate=0.1, random_state=42,
    #     n_iter_no_change=100, subsample=0.5,
    #     max_depth=4)

    # xgb_model = xgb.XGBRegressor(
    #     objective='reg:squarederror', subsample=0.8, random_state=42)
    # reg = GridSearchCV(xgb_model,
    #                    {'max_depth': [2, 3, 4, 5, 6, 7, 8],
    #                     'n_estimators': [20, 50, 200, 500, 1000, 10000],
    #                     'learning_rate': [0.5, 0.2, 0.1, 0.01, 0.001]},
    #                    cv=5,
    #                    verbose=1)

    cv = LeaveOneGroupOut()

    scores = -cross_val_score(reg,
                              X,
                              y, cv=cv, groups=fold_idx,
                              scoring='neg_mean_absolute_error',
                              n_jobs=-1)

    print('MAE : %s (+/- %s)' % (np.mean(scores), np.std(scores)))
    regression_scores[key] = scores


regression_scores.to_csv(SCORES, index=False)
