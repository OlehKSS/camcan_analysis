"""Plot mean absolute error (MAE) versus fMRI timeseries duration."""
from itertools import product
import os

import matplotlib.pyplot as plt
import pandas as pd


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data_trunc_fmri.h5'
OUT_FTYPE = 'png'

all_regressions = pd.read_hdf(PREDICTIONS, key='regression')
# 520 = 8min 40s - max duration of fMRI session
all_regressions.loc[all_regressions.duration.isnull(), 'duration'] = 520
# Plot errors of predictions from different modalities versus subject's age
all_regressions.sort_values(by=['duration'])
atlases = all_regressions.atlas.unique()
connectivities = all_regressions.connect.unique()
durations = all_regressions.duration.unique()
name_map = {'basc197': 'BASC 197', 'modl256': 'MODL 256',
            'correlation': 'r2z', 'tangent': 'tan'}

plt.figure()

for a, c in product(atlases, connectivities):
    label = f'{name_map[a]} {name_map[c]}'
    data = all_regressions.loc[(all_regressions.connect == c) &
                               (all_regressions.atlas == a), 'MAE']
    plt.plot(durations, data, '.-', label=label)

title = f'Age Prediction MAE vs Timeseries Duration'
plt.title(title)
plt.xlabel('Timeseries Duration (Seconds)')
plt.ylabel('MAE (Years)')
plt.grid()
plt.legend()
name = f'age-vs-ts_duration.{OUT_FTYPE}'
plt.savefig(os.path.join(FIG_OUT_PATH, name), bbox_inches='tight')
