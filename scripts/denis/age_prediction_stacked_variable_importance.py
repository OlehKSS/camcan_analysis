"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from camcan.processing import permutation_importance

N_JOBS = 50
N_THREADS = 5
DROPNA = 'global'
N_REPEATS = 10

IN_PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'
OUT_IMPORTANCE = './data/age_stacked_importance_{}.csv'


data = pd.read_hdf(IN_PREDICTIONS, key='predictions')

# Add extra dfeatures
meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')[['alpha_peak']]
meg_peaks = pd.read_csv(MEG_PEAKS).set_index('subject')[['aud', 'vis']]
meg_peaks2 = pd.read_csv(MEG_PEAKS2).set_index('subject')[['audvis']]
meg_peaks.columns = ['MEG ' + cc for cc in meg_peaks.columns]
meg_peaks2.columns = ['MEG ' + cc for cc in meg_peaks2.columns]
meg_extra.columns = ['MEG ' + cc for cc in meg_extra.columns]

data = data.join(meg_extra).join(meg_peaks).join(meg_peaks2)

FREQ_BANDS = ('alpha',
              'beta_high',
              'beta_low',
              'delta',
              'gamma_high',
              'gamma_lo',
              'gamma_mid',
              'low',
              'theta')

meg_source_types = (
    'mne_power_diag',
    'mne_power_cross',
    'mne_envelope_diag',
    'mne_envelope_cross',
    'mne_envelope_corr',
    'mne_envelope_corr_orth'
)

all_connectivity = [f'MEG {tt} {fb}' for tt in meg_source_types
                    if 'diag' not in tt for fb in FREQ_BANDS]

power_by_freq = [f'MEG {tt} {fb}' for tt in meg_source_types
                 if 'diag' in tt and 'power' in tt for fb in FREQ_BANDS]
envelope_by_freq = [f'MEG {tt} {fb}' for tt in meg_source_types
                    if 'diag' in tt and 'envelope' in tt for fb in FREQ_BANDS]

envelope_cov = [f'MEG {tt} {fb}' for tt in meg_source_types
                if 'cross' in tt and 'envelope' in tt for fb in FREQ_BANDS]

power_cov = [f'MEG {tt} {fb}' for tt in meg_source_types
             if 'cross' in tt and 'power' in tt for fb in FREQ_BANDS]

stacked_keys = {
    'MEG power and envelope by freq': power_by_freq + envelope_by_freq,
    'connectivity': all_connectivity,
    'MEG power + connectivity': (power_by_freq +
                                 envelope_by_freq + all_connectivity),
    'MEG all (no  diag)': ({cc for cc in data.columns if 'MEG' in cc} -
                           set(power_by_freq) - set(envelope_by_freq)),
    'MEG all': list(data.columns)
}

MRI = ['Cortical Surface Area', 'Cortical Thickness', 'Subcortical Volumes',
       'Connectivity Matrix, MODL 256 tan']
stacked_keys['ALL'] = list(stacked_keys['MEG all']) + MRI


def run_importance(data, stacked_keys):
    all_results = dict()
    for key, sel in stacked_keys.items():
        this_data = data[sel]
        if DROPNA == 'local':
            mask = this_data.dropna().index
        elif DROPNA == 'global':
            mask = data.dropna().index
        else:
            mask = this_data.index
        X = this_data.loc[mask].values
        y = data['age'].loc[mask].values

        if DROPNA is False:
            # code missings to make the tress learn from it.
            X_left = X.copy()
            X_left[this_data.isna().values] = -1000
            X_right = X.copy()
            X_right[this_data.isna().values] = 1000
            assert np.sum(np.isnan(X_left)) == 0
            assert np.sum(np.isnan(X_right)) == 0
            assert np.min(X_left) == -1000
            assert np.max(X_right) == 1000
            X = np.concatenate([X_left, X_right], axis=1)

        n_estimators = 1000

        regs = [
            ('rf_msqrt',
             RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=N_JOBS,
                                   max_features='sqrt',
                                   max_depth=5,
                                   random_state=42)),
            ('rf_m1',
             RandomForestRegressor(n_estimators=n_estimators,
                                   max_features=1,
                                   max_depth=5,
                                   n_jobs=N_JOBS,
                                   random_state=42)),
            ('et_m1',
             ExtraTreesRegressor(n_estimators=n_estimators,
                                 max_features=1,
                                 max_depth=5,
                                 n_jobs=N_JOBS,
                                 random_state=42)),
        ]
        results = list()
        for mod_type, reg in regs:
            importance_result = pd.DataFrame(
                columns=sel,
                index=range(n_estimators + 1))
            reg.fit(X, y)
            importance_result.loc[0] = reg.feature_importances_
            for ii, tree in enumerate(reg.estimators_, 1):
                importance_result.loc[ii] = tree.feature_importances_
            importance_result['mod_type'] = mod_type
            results.append(importance_result)
        results = pd.concat(results, axis=0)

        n_permuations = 100
        importance_result = pd.DataFrame(
            columns=sel,
            index=range(n_permuations))
        estimator = regs[0][1]
        permutation_result = permutation_importance(
            estimator=estimator, X=X, y=y,
            n_repeats=n_permuations,
            n_jobs=N_JOBS,
            scoring='neg_mean_absolute_error')
        importance = permutation_result['importances'].T
        for ii in range(n_permuations):
            importance_result.loc[ii] = importance[ii]
        importance_result['mod_type'] = 'permutation'
        results = results.append(importance_result)
        all_results[key] = results

    return all_results


DEBUG = False
if DEBUG:
    N_JOBS = 1
    data = data.iloc[::6]

data = data.query("repeat == 0")

out = run_importance(data, {k: v for k, v in stacked_keys.items()
                            if k == 'MEG power and envelope by freq'})
for key, val in out.items():
    val.to_hdf(OUT_IMPORTANCE.format(key), key='importance')
