import pandas as pd

from camcan.utils import run_ridge, plot_pred, plot_learning_curve, plot_barchart, StackingRegressor

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import (cross_val_score, cross_val_predict, learning_curve, ShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


CV = 10
# store mae, std for the summary plot
mae_std = {}
subjects_data = pd.read_csv('/home/okozyn/Projects/inria/camcan_analysis/data/participant_data.csv', index_col=0)
area_data = pd.read_hdf('/home/okozyn/Projects/inria/camcan_analysis/data/structural/structural_data.h5', 
                           key='area')
thickness_data = pd.read_hdf('/home/okozyn/Projects/inria/camcan_analysis/data/structural/structural_data.h5',
                               key='thickness')
volume_data = pd.read_hdf('/home/okozyn/Projects/inria/camcan_analysis/data/structural/structural_data.h5',
                            key='volume')

from sklearn.compose import ColumnTransformer

# ********** delete it later ******** 
alphas = None
train_sizes = None
data = pd.concat([area_data, thickness_data, volume_data], axis=1, join='inner')
cv = CV
# ***********************************


if alphas is None:
    alphas = np.logspace(start=-3, stop=1, num=50, base=10.0)
if train_sizes is None:
    train_sizes = np.linspace(.1, 1.0, 5)
    
_, area_col = area_data.shape
_, thickness_col = thickness_data.shape
_, volume_col = volume_data.shape

# prepare data, subjects age
data_rnd = data.sample(frac=1)
subjects = data_rnd.index.values
y = subjects_data.loc[data_rnd.index.values].age.values
X = data_rnd.values

cv_ss = ShuffleSplit(n_splits=cv, random_state=42)

ct_area = ColumnTransformer([('pass_area', 'passthrough', slice(0, area_col)),
                             ('drop_thickness', 'drop', slice(area_col, area_col + thickness_col)),
                             ('drop_volume', 'drop', slice(area_col + thickness_col,
                                                           area_col + thickness_col + volume_col))])
ct_thickness = ColumnTransformer([('drop_area', 'drop', slice(0, area_col)),
                                  ('pass_thickness', 'passthrough', slice(area_col, area_col + thickness_col)),
                                  ('drop_volume', 'drop', slice(area_col + thickness_col,
                                                                area_col + thickness_col + volume_col))])
ct_volume = ColumnTransformer([('drop_area', 'drop', slice(0, area_col)),
                               ('drop_thickness', 'drop', slice(area_col, area_col + thickness_col)),
                               ('pass_volume', 'passthrough', slice(area_col + thickness_col,
                                                                    area_col + thickness_col + volume_col))])
# RidgeCV(alphas)
estimators = [
    ('reg_area', make_pipeline(ct_area, StandardScaler(), Ridge(0.5))),
    ('reg_thickness', make_pipeline(ct_thickness, StandardScaler(), Ridge(0.5))),
    ('reg_volume', make_pipeline(ct_volume, StandardScaler(), Ridge(0.5)))
]

clf = StackingRegressor(estimators=estimators,
                        final_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                        cv=cv_ss, random_state=42)


# mae = cross_val_score(clf, X, y, scoring='neg_mean_absolute_error', cv=cv_ss)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

out = clf.fit(X_train, y_train).score(X_test, y_test)
# r2 = cross_val_score(clf, X, y, scoring='r2', cv=cv_ss)
# y_pred = cross_val_predict(clf, X, y, cv=cv_ss)

# train_sizes, train_scores, test_scores = \
#     learning_curve(reg, X, y, cv=cv, train_sizes=train_sizes, scoring="neg_mean_absolute_error")