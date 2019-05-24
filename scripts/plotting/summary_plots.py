import os
import pickle as pkl
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'png'

out_folder = os.path.join(FIG_OUT_PATH, 'boxplots')

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

all_regressions = pd.read_hdf(PREDICTIONS, key='regression')
all_r2_scores = pd.read_hdf(PREDICTIONS, key='r2')

# Plot summary boxplot
keys = all_regressions.index

# have I done boxplots right?

title = 'Age Predictions, Combined'
sns.set_style('darkgrid')
plt.figure()
ax = sns.boxplot(data=all_regressions.transpose(),
                 showmeans=True,
                 orient='h')
ax.set_title(title)
ax.set(xlabel='Absolute Prediction Error (Years)')

name = f'combined_plot.{OUT_FTYPE}'
plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')

# save mae, std to a csv file
summary = pd.DataFrame(index=all_regressions.index,
                       columns=['mae', 'std'], dtype=float)
summary['mae'] = all_regressions.mean(axis=1)
summary['std'] = all_regressions.std(axis=1)
summary['r2'] = all_r2_scores.mean(axis=1)
summary.to_csv('../../data/summary.csv')
