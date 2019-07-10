"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

PREDICTIONS = '../../data/age_prediction_exp_data.h5'

data = pd.read_hdf(PREDICTIONS, key='predictions')
data = data.dropna()

y = data['age']
fold_idx = data['fold_idx']
data = data.drop(['fold_idx', 'age'], axis=1)
X = data.values

reg = RandomForestRegressor(n_estimators=300, max_depth=5)
cv = LeaveOneGroupOut()

scores = -cross_val_score(reg, X, y, cv=cv, groups=fold_idx,
                          scoring='neg_mean_absolute_error',
                          n_jobs=-1)

print('MAE : %s (+/- %s)' % (np.mean(scores), np.std(scores)))
