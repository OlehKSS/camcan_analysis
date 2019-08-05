"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import partial_dependence
from mne.externals import h5io

N_JOBS = 40
N_THREADS = 1
DROPNA = 'global'
N_REPEATS = 10
IN_PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'
OUT_DEPENDENCE = './data/age_stacked_dependence_{}.h5'


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
    'MEG-all-no-diag': ({cc for cc in data.columns if 'MEG' in cc} -
                        {'MEG envelope diag', 'MEG power diag'}),
    'MEG all': [cc for cc in data.columns if 'MEG' in cc]
}

MRI = ['Cortical Surface Area', 'Cortical Thickness', 'Subcortical Volumes',
       'Connectivity Matrix, MODL 256 tan']
stacked_keys['ALL'] = list(stacked_keys['MEG all']) + MRI

#  we put in here keys ordered by importance.
dependence_map = {
    'MEG all': {'1d': ['MEG envelope diag',
                       'MEG power diag',
                       'MEG mne_envelope_cross alpha',
                       'MEG mne_envelope_cross beta_low',
                       'MEG mne_power_diag beta_low',
                       'MEG mne_envelope_diag beta_low',
                       'MEG mne_envelope_cross theta',
                       'MEG mne_envelope_corr alpha',
                       'MEG mne_envelope_corr beta_low',
                       'MEG mne_power_cross beta_high']}
}
dependence_map['MEG all']['2d'] = list(
    combinations(dependence_map['MEG all']['1d'][:4], 2))


def run_dependence(data, stacked_keys, dependence_map):
    all_results = list()
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
        # For each set of stacked modesl,
        # we fit the model variants we used for importance computing.
        for mod_type, reg in regs:
            reg.fit(X, y)
            # first we compute the 1d dependence for this configuration
            pdp_output = {'1d': dict(), '2d': dict(),
                          'mod_type': mod_type, 'stack_model': key}
            for var_1d in dependence_map[key]['1d']:
                print(var_1d)
                # idea: bootstrap predictions by subsamping tress and
                # hacking fitted objects here the estimator list is
                # overwritten with bootstraps.
                pdp_output['1d'][var_1d] = partial_dependence(
                    estimator=reg,
                    X=X,
                    ci=(0, 1),
                    features=[this_data.columns.tolist().index(var_1d)])
            for vars_2d in dependence_map[key]['2d']:
                print(vars_2d)
                # idea: bootstrap predictions by subsamping tress and
                # hacking fitted objects here the estimator list is
                # overwritten with bootstraps.
                feat_idx = [this_data.columns.tolist().index(vv)
                            for vv in vars_2d]
                pdp_output['2d']['-'.join(vars_2d)] = partial_dependence(
                    estimator=reg,
                    X=X,
                    # ci=(0, 1),
                    features=[feat_idx])


            all_results.append(pdp_output)

    return all_results


DEBUG = True
if DEBUG:
    N_JOBS = 1
    data = data.iloc[::6]
    stacked_keys = {k: v for k, v in stacked_keys.items()
                    if k == 'MEG all'}

out = run_dependence(data, stacked_keys, dependence_map)

h5io.write_hdf5(
    OUT_DEPENDENCE.format('model-full'),  out, compression=9,
    overwrite=True)