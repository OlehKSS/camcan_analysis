import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import (cross_val_score,
                                        cross_val_predict,
                                        learning_curve,
                                        ShuffleSplit, check_cv,
                                        LeaveOneGroupOut)
from sklearn.ensemble import RandomForestRegressor

alphas = np.logspace(-3, 5, 100)

CV = 10
N_JOBS = 2
PANDAS_OUT_FILE = '../../data/age_prediction_exp_data_denis.h5'
STRUCTURAL_DATA = '../../data/structural/structural_data.h5'
CONNECT_DATA_CORR = '../../data/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = '../../data/connectivity/connect_data_tangent.h5'
MEG_SOURCE_SPACE_DATA = '../../data/meg_source_space_data.h5'
MEG_EXTRA_DATA = '../../data/meg_extra_data.h5'
MEG_PEAKS = '../../data/evoked_peaks.csv'
MEG_ENV_CORR = '../../data/all_power_envelopes.h5'

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

columns_to_exclude = ('band', 'fmax', 'fmin', 'subject')
parcellation_labels = [c for c in meg_data.columns if c
                       not in columns_to_exclude]
band_data = [meg_data[meg_data.band == bb].set_index('subject')[
                parcellation_labels] for bb in FREQ_BANDS]
meg_data = pd.concat(band_data, axis=1, join='inner', sort=False)
meg_subjects = set(meg_data.index)

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
meg_data = meg_data.loc[common_subjects]

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


# prepare data, subjects age
subjects = common_subjects
y = subjects_data.loc[common_subjects].age.values
X = meg_data.values
cv = KFold(n_splits=CV, shuffle=True, random_state=42)
cv = check_cv(cv)

out = pd.DataFrame(index=common_subjects)
out['age'] = subjects_data.loc[out.index].age

rnd_state = 42
ada_n_estimators = 10

for fold_idx, (train, test) in enumerate(cv.split(X)):
    ridge_reg = make_pipeline(StandardScaler(), RidgeCV(alphas))
    ada_reg = AdaBoostRegressor(RidgeCV(alphas), n_estimators=ada_n_estimators,
                                random_state=42)
    ada_reg = make_pipeline(StandardScaler(), ada_reg)
    ridge_reg.fit(X[train], y[train])
    ada_reg.fit(X[train], y[train])

    out.loc[out.index[test], 'ridge'] = ridge_reg.predict(X[test])
    out.loc[out.index[test], 'fold_idx'] = fold_idx

    for index, est in enumerate(ada_reg.steps[-1][-1].estimators_):
        out.loc[out.index[test], f'ada{index}'] =\
            est.predict(ada_reg.steps[0][-1].transform(X[test]))

print(f'Ridge MAE {np.sum(np.abs(out.ridge - out.age)) / len(out)}')
print(f'Ada0 MAE {np.sum(np.abs(out.ada0 - out.age)) / len(out)}')

y = out['age']
fold_idx = out['fold_idx']
X = out.drop(['fold_idx', 'age'], axis=1)
X = X.values

reg = RandomForestRegressor(n_estimators=500, max_depth=5,
                            random_state=42)
cv = LeaveOneGroupOut()

scores = -cross_val_score(reg, X, y, cv=cv, groups=fold_idx,
                          scoring='neg_mean_absolute_error', n_jobs=-1)

print(f'RF MAE {np.mean(scores)} (+/- {np.std(scores)})')
