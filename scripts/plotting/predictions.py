import os
import pickle as pkl
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'png'

out_folder = os.path.join(FIG_OUT_PATH, 'predictions')

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

all_predictions = pd.read_hdf(PREDICTIONS, key='predictions')
all_regressions = pd.read_hdf(PREDICTIONS, key='regression')
# Plot errors of predictions from different modalities versus subject's age
keys = all_predictions.columns
# remove column with the original age
keys = keys[1:]

for key in keys:
    title = f'Age Prediction, {key}'
    slice_pred = all_predictions[key].dropna()
    y = all_predictions.loc[slice_pred.index].age
    y_pred = slice_pred.values
    reg_eval = all_regressions.loc[key].values
    mae = reg_eval.mean()
    plt.close()
    plt.figure()
    plt.title(title)
    plt.scatter(y, y_pred,  edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=3, color='green')
    plt.plot([y.min(), y.max()], [y.min() - mae, y.max() - mae], 'k--', lw=3, color='red')
    plt.plot([y.min(), y.max()], [y.min() + mae, y.max() + mae], 'k--', lw=3, color='red')
    plt.xlabel('Chronological Age (Years)')
    plt.ylabel('Predicted Age (Years)')
    plt.grid()

    name = f'age_pred_{key.replace(" ", "-")}.{OUT_FTYPE}'
    plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')
