"""Age prediction using MRI, fMRI and MEG data."""
import os.path as op
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Memory

from mne.externals import h5io

from camcan.utils import (run_stacking, run_ridge)
from threadpoolctl import threadpool_limits
from camcan.processing import map_tangent

##############################################################################
# Paths

DRAGO_PATH = '/storage/inria/agramfor/camcan_derivatives'
OLEH_PATH = '/storage/tompouce/okozynet/projects/camcan_analysis/data'
PANDAS_OUT_FILE = './data/age_prediction_exp_data_denis.h5'
STRUCTURAL_DATA = f'{OLEH_PATH}/structural/structural_data.h5'
CONNECT_DATA_CORR = f'{OLEH_PATH}/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = f'{OLEH_PATH}/connectivity/connect_data_tangent.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'

##############################################################################
# Control paramaters

# common subjects 574
CV = 10
N_JOBS = 40
memory = Memory(location=DRAGO_PATH)

##############################################################################
# Subject info

# read information about subjects
subjects_data = pd.read_csv('./data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age,
                                    index=subjects_data.index,
                                    dtype=float)

##############################################################################
# MEG features
#
# 1. Marginal Power
# 2. Cross-Power
# 3. Envelope Power
# 4. Envelope Cross-Power
# 5. Envelope Connectivity
# 6. Envelope Orthogonalized Connectivity
# 7. 1/f
# 8. Alpha peak
# 9. ERF delay

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

def vec_to_sym(data, n_rows, skip_diag=True):
    """Put vector back in matrix form"""
    if skip_diag:
        k = 1
        # This is usually true as we write explicitly
        # the diag info in asecond step and we only
        # store the upper triangle, hence all files
        # have equal size.
    else:
        k = 0
    C = np.zeros((n_rows, n_rows), dtype=np.float64)
    C[np.triu_indices(n=n_rows, k=k)] = data
    C += C.T
    if not skip_diag:
        C.flat[::n_rows + 1] = np.diag(C) / 2.
    return C

def make_covs(diag, data, n_labels):
    if not np.isscalar(diag):
        assert np.all(diag.index == data.index)
    covs = np.empty(shape=(len(data), n_labels, n_labels))
    for ii, this_cross in enumerate(data.values):
        C = vec_to_sym(this_cross, n_labels)
        if np.isscalar(diag):
            this_diag = diag
        else:
            this_diag = diag.values[ii]
        C.flat[::n_labels + 1] = this_diag
        covs[ii] = C
    return covs

@memory.cache
def read_meg_rest_data(kind, band, n_labels=448):
    """Read the resting state data (600 subjects)
    
    Read connectivity outptus and do some additional
    preprocessing.

    Parameters
    ----------
    kind : str
        The type of MEG feature.
    band : str
        The frequency band.
    n_label: int
        The number of ROIs in source space.
    """
    if kind == 'mne_power_diag':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_source_power_diag-{band}.h5'),
            key=kind)
    elif kind == 'mne_power_cross':
        # We need the diagonal powers to do tangent mapping.
        # but then we will discard it.
        diag = read_meg_rest_data(kind='mne_power_diag', band=band)
        # undp log10
        diag = diag.transform(lambda x: 10 ** x)
        index = diag.index.copy()
    
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_source_power_cross-{band}.h5'),
            key=kind)
        covs = make_covs(diag, data, n_labels)
        data = map_tangent(covs, diag=True)
        data = pd.DataFrame(data=data, index=index)
    if kind == 'mne_envelope_diag':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_diag_{band}.h5'),
            key=kind)
    elif kind == 'mne_envelope_cross':
        # We need the diagonal powers to do tangent mapping.
        # but then we will discard it.
        diag = read_meg_rest_data(kind='mne_envelope_diag', band=band)
        # undp log10
        diag = diag.transform(lambda x: 10 ** x)
        index = diag.index.copy()

        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_cross_{band}.h5'),
            key=kind)
        covs = make_covs(diag, data, n_labels)
        data = map_tangent(covs, diag=True)
        data = pd.DataFrame(data=data, index=index)
    elif kind == 'mne_envelope_corr':
        # The diagonal is simply one.
        diag = 1.0
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_corr_{band}.h5'),
            key=kind)
        index = data.index.copy()

        data = map_tangent(make_covs(diag, data, n_labels),
                           diag=True)
        data = pd.DataFrame(data=data, index=index)

    elif kind == 'mne_envelope_corr_orth':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_corr_orth_{band}.h5'),
            key=kind)
        # The result here is not an SPD matrix.
        # We do do Fisher's Z-transform instead.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        data = data.transform(np.arctanh)
    return data

meg_power_alpha = read_meg_rest_data(
    kind='mne_power_diag', band='alpha')

meg_subjects = set(meg_power_alpha.index)

meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')

meg_peaks = pd.read_csv(MEG_PEAKS).set_index('subject')

meg_subjects = (meg_subjects.intersection(meg_extra.index)
                            .intersection(meg_peaks.index))

##############################################################################
# MRI features

area_data = pd.read_hdf(STRUCTURAL_DATA, key='area')
thickness_data = pd.read_hdf(STRUCTURAL_DATA, key='thickness')
volume_data = pd.read_hdf(STRUCTURAL_DATA, key='volume')

area_data = area_data.dropna()
thickness_data = thickness_data.dropna()
volume_data = volume_data.dropna()

# take only subjects that are both in MEG and Structural MRI
structural_subjects = set(area_data.index)

##############################################################################
# Bundle all data

common_subjects = list(meg_subjects.intersection(structural_subjects))
common_subjects.sort()

area_data = area_data.loc[common_subjects]
thickness_data = thickness_data.loc[common_subjects]
volume_data = volume_data.loc[common_subjects]

meg_extra = meg_extra.loc[common_subjects]
meg_peaks = meg_peaks.loc[common_subjects]

# read connectivity data
connect_data_tangent_modl = pd.read_hdf(CONNECT_DATA_TAN, key='modl256')

# use only common subjects
connect_data_tangent_modl = connect_data_tangent_modl.loc[common_subjects]

print('Data was read successfully.')

data_ref = {
    'MEG 1/f low': meg_extra[
        [cc for cc in meg_extra.columns if '1f_low' in cc]],
    'MEG 1/f gamma': meg_extra[
        [cc for cc in meg_extra.columns if '1f_gamma' in cc]],
    'MEG 1/f gamma': meg_extra[
        [cc for cc in meg_extra.columns if '1f_gamma' in cc]],
    'Cortical Surface Area': area_data,
    'Cortical Thickness': thickness_data,
    'Subcortical Volumes': volume_data,
    'Connectivity Matrix, MODL 256 tan': connect_data_tangent_modl,
}
for band in FREQ_BANDS:
    for kind in meg_source_types:
        data_ref[f"MEG {kind} {band}"] = dict(kind=kind, band=band)

##############################################################################
# Prepare outputs

# store mae, learning curves for summary plots
regression_mae = pd.DataFrame(columns=range(0, CV), dtype=float)
regression_r2 = pd.DataFrame(columns=range(0, CV), dtype=float)
learning_curves = {}

##############################################################################
# Main analysis

cv = KFold(n_splits=CV, shuffle=True, random_state=42)
with threadpool_limits(limits=N_JOBS, user_api='blas'):
    for key, data in data_ref.items():
        if isinstance(data, dict):
            data = read_meg_rest_data(**data)

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
