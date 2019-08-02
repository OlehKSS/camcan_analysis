"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

N_REPEATS = 10
DROPNA = 'global'

PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'
SCORES = './data/age_stacked_scores_{}.csv'
PREDICTIONS = './data/age_stacked_predictions_{}.csv'


data = pd.read_hdf(PREDICTIONS, key='predictions')

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

meg_high_level = [
    'MEG power diag',
    'MEG envelope diag',
    'MEG alpha_peak',
    'MEG 1/f low',
    'MEG 1/f gamma',
    'MEG aud',
    'MEG vis',
    'MEG audvis'
]

stacked_keys = {
    'MEG power': ['MEG power diag'],
    'MEG high-level': meg_high_level,
    'MEG connectivity': all_connectivity,
    'MEG high-level + connectivity': meg_high_level + all_connectivity,
    'MEG high-level + cov': meg_high_level + envelope_cov + power_cov,
    'MEG high-level + cov (no env)': [c for c in (meg_high_level + power_cov)
                                      if 'envelope' not in c],
    'MEG power by freq': power_by_freq,
    'MEG envelope by freq': envelope_by_freq,
    'MEG power and envelope by freq': power_by_freq + envelope_by_freq,
    'MEG handcrafted': meg_high_level[4:],
    'MEG frequency-resloved': (meg_high_level[2:] + power_by_freq +
                               envelope_by_freq),
    'MEG frequency-resloved + connectivity': (meg_high_level[2:] +
                                              power_by_freq +
                                              envelope_by_freq + envelope_cov),
    'MEG all': ({cc for cc in data.columns
                 if 'MEG' in cc} - set(power_by_freq)) - set(envelope_by_freq)
}

MRI = ['Cortical Surface Area', 'Cortical Thickness', 'Subcortical Volumes',
       'Connectivity Matrix, MODL 256 tan']
stacked_keys['ALL'] = list(stacked_keys['MEG all']) + MRI
stacked_keys['ALL no fMRI'] = list(stacked_keys['MEG all']) + MRI[:-1]
stacked_keys['MRI'] = MRI[:-1]
stacked_keys['ALL MRI'] = MRI


def get_mae(predictions, key):
    scores = []
    for fold_idx, df in predictions.groupby('fold_idx'):
        scores.append(np.mean(np.abs(df[key] - df['age'])))
    return scores


def fit_predict_score(estimator, X, y, train, test):
    estimator.fit(X[train], y[train])
    y_pred = estimator.predict(X[test])
    score_mae = mean_absolute_error(y_true=y[test], y_pred=y_pred)
    return (y_pred, score_mae)


def run_stacked(data, stacked_keys, repeat_idx):
    out_scores = pd.DataFrame()
    out_predictions = data.copy()
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
        fold_idx = data.loc[mask]['fold_idx'].values

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

        for column in sel:
            score = get_mae(data.loc[mask], column)
            if column not in out_scores:
                out_scores[column] = score
            elif out_scores[column].mean() < np.mean(score):
                out_scores[column] = score

        unstacked = out_scores[sel].values
        idx = unstacked.mean(axis=0).argmin()
        unstacked_mean = unstacked[:, idx].mean()
        unstacked_std = unstacked[:, idx].std()
        print(f'{key} | best unstacked MAE: {unstacked_mean} '
              f'(+/- {unstacked_std}')

        print('n =', len(X))

        param_grid = {'max_depth': [4, 6, 8, None]}
        if X.shape[1] > 10:
            param_grid['max_features'] = (['log2', 'sqrt', None])

        reg = GridSearchCV(
            RandomForestRegressor(n_estimators=1000,
                                  random_state=42),
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            iid=False,
            cv=5)

        cv = LeaveOneGroupOut()
        out_cv = Parallel(n_jobs=1)(delayed(fit_predict_score)(
            estimator=reg, X=X, y=y, train=train, test=test)
            for train, test in cv.split(X, y, fold_idx))

        out_cv = zip(*out_cv)
        predictions = np.concatenate(next(out_cv), axis=0)
        scores = np.array(next(out_cv))
        out_predictions[f'stacked_{key}'] = np.nan
        out_predictions.loc[mask, f'stacked_{key}'] = predictions
        print(f'{key} | MAE : %s (+/- %s)' % (np.mean(scores), np.std(scores)))
        out_scores[key] = scores
    out_scores['repeat_idx'] = repeat_idx
    out_predictions['repeat_idx'] = repeat_idx
    return out_scores, out_predictions


out = Parallel(n_jobs=10)(delayed(run_stacked)(
    data.query(f"repeat == {ii}"), stacked_keys, ii)
    for ii in range(N_REPEATS))
out = zip(*out)

out_scores_meg = next(out)
out_scores_meg.to_csv(
    SCORES.format('meg' + DROPNA if DROPNA else '_na_coded'),
    index=False)

out_predictions_meg = next(out)
out_predictions_meg.to_csv(
    PREDICTIONS.format('meg' + DROPNA if DROPNA else '_na_coded'),
    index=False)
