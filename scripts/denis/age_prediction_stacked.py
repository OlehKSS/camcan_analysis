"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
import os.path as op
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, LeaveOneGroupOut, ShuffleSplit)

PREDICTIONS = './data/age_prediction_exp_data_na_denis.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'
SCORES = './data/age_prediction_scores_{}.csv'

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
    'MEG frequency-resloved + connectivity': (meg_high_level[2:] + power_by_freq +
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


DROPNA = True


def get_mae(predictions, key):
    scores = []
    for fold_idx, df in predictions.groupby('fold_idx'):
        scores.append(np.mean(np.abs(df[key] - df['age'])))
    return scores


def run_stacked(data, stacked_keys):
    regression_scores = pd.DataFrame()
    for key, sel in stacked_keys.items():
        this_data = data[sel]
        if DROPNA:
            mask = this_data.dropna().index
        else:
            mask = Ellipsis
        X = this_data.loc[mask].values
        y = data['age'].loc[mask].values
        fold_idx = data.loc[mask]['fold_idx'].values

        unstacked_mae = [get_mae(data.loc[mask], s) for s in sel]
        unstacked_mean = min(np.mean(x) for x in unstacked_mae)
        unstacked_std = min(np.std(x) for x in unstacked_mae)
        print(f'{key} | best unstacked MAE: {unstacked_mean} '
              f'(+/- {unstacked_std}')
        # redefine model
        print('n =', len(X))
        reg = GridSearchCV(
            RandomForestRegressor(n_estimators=1000,
                                  random_state=42),
            param_grid={'max_features': (['log2', 'sqrt', None]),
                        'max_depth': [4, 6, 8, None]},
            scoring='neg_mean_absolute_error',
            iid=False,
            cv=5)

        cv = LeaveOneGroupOut()
        scores = -cross_val_score(reg,
                                  X,
                                  y, cv=cv,
                                  groups=fold_idx,
                                  scoring='neg_mean_absolute_error',
                                  n_jobs=4)

        print(f'{key} | MAE : %s (+/- %s)' % (np.mean(scores), np.std(scores)))
        regression_scores[key] = scores
    return regression_scores

regression_scores_meg = run_stacked(data, stacked_keys)
regression_scores_meg.to_csv(SCORES.format('meg'), index=False)
