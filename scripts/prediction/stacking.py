import os

import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from camcan.processing import SPoC
from camcan.utils import (plot_pred, plot_learning_curve,
                          plot_barchart, run_meg_ridge,
                          plot_error_age, plot_error_segments,
                          plot_boxplot, run_stacking)

# How to do memory profiling
# https://pypi.org/project/memory-profiler/
# for time-memory profile mprof run <executable>
# Line profiler
# https://github.com/rkern/line_profiler

CV = 10
# store mae, std for the summary plot
meg_mae = {}
meg_mae_std = {}
meg_pred_diff = {}

subjects_data = pd.read_csv('../../data/participant_data.csv', index_col=0)
subjects_data.head()


# 643 subjects, each covariance is 9x306x306
meg_data = mne.externals.h5io.read_hdf5('../../data/covs_allch_oas.h5')

print(f'Found {len(meg_data)} subjects')
print(f'A covarince matrix shape is {meg_data[0]["covs"].shape}')


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
info = raw.info

# ## SPoC and Age Prediction

FREQ_BANDS = [(0.1, 1.5),  # low
              (1.5, 4.0),  # delta
              (4.0, 8.0),  # theta
              (8.0, 15.0),  # alpha
              (15.0, 26.0),  # beta_low
              (26.0, 35.0),  # beta_high
              (35.0, 50.0),  # gamma_low
              (50.0, 74.0),  # gamma_mid
              (76.0, 120.0)]  # gamma_high

# read data of other predictors
area_data = pd.read_hdf('../../data/structural/structural_data.h5', 
                           key='area')
thickness_data = pd.read_hdf('../../data/structural/structural_data.h5',
                               key='thickness')
volume_data = pd.read_hdf('../../data/structural/structural_data.h5',
                            key='volume')

area_data = area_data.dropna()
thickness_data = thickness_data.dropna()
volume_data = volume_data.dropna()

connect_data_tangent_basc = pd.read_hdf('../../data/connectivity/connect_data_tangent.h5',
                              key='basc197')
connect_data_tangent_modl = pd.read_hdf('../../data/connectivity/connect_data_tangent.h5',
                              key='modl256')


multimodal_data = [('area', area_data), ('thickness', thickness_data), ('volume', volume_data),
                   ('basc', connect_data_tangent_basc), ('modl', connect_data_tangent_modl),
                   ('meg', meg_data)]

df_pred, arr_mae, r2, train_sizes, train_scores, test_scores\
    = run_stacking(multimodal_data, subjects_data, cv=CV, fbands=FREQ_BANDS)

arr_mae = -arr_mae
mae = arr_mae.mean()
std = arr_mae.std()

print('MAE: %.2f' % mae)
print('MAE STD: %.2f' % std)
