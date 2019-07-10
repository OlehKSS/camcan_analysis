"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data_small.h5'
SCORES = '../../data/age_prediction_scores.csv'

data = pd.read_hdf(PREDICTIONS, key='predictions')
data = data.dropna()

stacked_keys = {
    'MEG, Cortical Surface Area Stacked-multimodal': ['Cortical Surface Area',
                                                      'MEG'],
    'MEG, Cortical Thickness Stacked-multimodal': ['Cortical Thickness', 'MEG'],
    'MEG, Subcortical Volumes Stacked-multimodal': ['Subcortical Volumes', 'MEG'],
    'MEG, BASC 197 tan Stacked-multimodal': ['Connectivity Matrix, BASC 197 tan',
                                             'MEG'],
    'MEG, MODL 256 r2z Stacked-multimodal': ['Connectivity Matrix, MODL 256 r2z',
                                             'MEG'],
    'MRI Stacked': ['Cortical Surface Area', 'Cortical Thickness',
                    'Subcortical Volumes'],
    'fMRI Stacked': ['Connectivity Matrix, BASC 197 tan',
                     'Connectivity Matrix, MODL 256 r2z'],
    'MRI, fMRI Stacked-multimodal': ['Cortical Surface Area',
                                     'Cortical Thickness',
                                     'Subcortical Volumes',
                                     'Connectivity Matrix, BASC 197 tan'],
    'MEG, MRI Stacked-multimodal': ['Cortical Surface Area',
                                    'Cortical Thickness',
                                    'Subcortical Volumes',
                                    'MEG'],
    'MEG, fMRI Stacked-multimodal': ['Connectivity Matrix, BASC 197 tan',
                                     'MEG'],
    'MEG, MRI, fMRI Stacked-multimodal': ['Cortical Surface Area',
                                          'Cortical Thickness',
                                          'Subcortical Volumes',
                                          'Connectivity Matrix, BASC 197 tan',
                                          'MEG']
}

key_labels = {'Cortical Surface Area': 'Cortical Surface Area',
              'Cortical Thickness': 'Cortical Thickness',
              'Subcortical Volumes': 'Subcortical Volumes',
              'Connectivity Matrix, BASC 197 tan': 'BASC 197 tan',
              'Connectivity Matrix, MODL 256 tan': 'MODL 256 tan',
              'Connectivity Matrix, MODL 256 r2z': 'MODL 256 r2z',
              'Connectivity Matrix, BASC 197 r2z': 'BASC 197 r2z',
              'MEG': 'MEG'
              }


def get_mae(predictions, key):
    scores = []
    for fold_idx, df in predictions.groupby('fold_idx'):
        scores.append(np.mean(np.abs(df[key] - df['age'])))
    return scores


regression_scores = pd.DataFrame()

for key in key_labels:
    regression_scores[key] = get_mae(data, key)

# Do the stacking:

y = data['age']
fold_idx = data['fold_idx']
data = data.drop(['fold_idx', 'age'], axis=1)

for key, val in stacked_keys.items():
    X = data[val].values

    reg = RandomForestRegressor(n_estimators=300, max_depth=5,
                                random_state=42)
    # reg = GradientBoostingRegressor(n_estimators=300, max_depth=5)
    cv = LeaveOneGroupOut()

    scores = -cross_val_score(reg, X, y, cv=cv, groups=fold_idx,
                              scoring='neg_mean_absolute_error',
                              n_jobs=-1)

    print('MAE : %s (+/- %s)' % (np.mean(scores), np.std(scores)))
    regression_scores[key] = scores


regression_scores.to_csv(SCORES, index=False)
