from itertools import combinations
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'png'

data = pd.read_hdf(PREDICTIONS, key='predictions')
# Plot errors of predictions from different modalities versus subject's age
keys = data.columns
# remove column with the original age
keys = keys[1:]

ylim = (-2, 55)
xlim = None
out_folder = os.path.join(FIG_OUT_PATH, 'ae_vs_age')

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

for key1 in keys:
    data_slice = data[key1].dropna()
    age = data.loc[data_slice.index, 'age'].values
    abs_errors = np.abs(data_slice.values - age)
    plt.close()
    plt.figure()
    plt.scatter(age, abs_errors, edgecolors='black')
    plt.title(key1)
    plt.xlabel('Age (Years)')
    plt.ylabel('Absolute Error (Years)')
    plt.grid()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    name = f'AE_vs_age_{key1.replace(" ", "-")}.{OUT_FTYPE}'
    plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')

# Plot errors of predictions from different modalities versus each other
data = data.dropna()
age = data.age.values
keys = data.columns
# remove column with the original age
keys = keys[1:]

xlim = (0, 55)
ylim = (0, 55)
out_folder = os.path.join(FIG_OUT_PATH, 'ae_predictor_vs_predictor')

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

title = 'Absolute Error Correlation'
for key1, key2 in combinations(keys, r=2):
    plt.close()
    fig, ax = plt.subplots()
    x_values = np.abs(data[key1].values - age)
    y_values = np.abs(data[key2].values - age)
    plt.scatter(x_values, y_values, edgecolors='black', c=age, cmap=plt.cm.viridis_r)
    plt.title(title)
    plt.xlabel(key1 + ', AE (Years)')
    plt.ylabel(key2 + ', AE (Years)')

    if xlim is not None:
        xlim_ = (xlim[0] - 1, xlim[1] + 1)
    else:
        xlim_ = (data[key1].min() - 1, data[key1].max() + 1)
    
    if ylim is not None:
        ylim_ = (ylim[0] - 1, ylim[1] + 1)
    else:
        ylim_ = (data[key2].min() - 1, data[key2].max() + 1)

    ax.set(xlim=xlim_, ylim=ylim_)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3')
    plt.grid()
    plt.colorbar()

    name = f'AE_{key1.replace(" ", "-")}_vs_{key2.replace(" ", "-")}.{OUT_FTYPE}'
    plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')
