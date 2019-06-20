import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'png'

out_folder = os.path.join(FIG_OUT_PATH, 'boxplots')

all_regressions = pd.read_hdf(PREDICTIONS, key='regression')
all_r2_scores = pd.read_hdf(PREDICTIONS, key='r2')

# save mae, std to a csv file
summary = pd.DataFrame(index=all_regressions.index,
                       columns=['mae', 'std'], dtype=float)
summary['mae'] = all_regressions.mean(axis=1)
summary['std'] = all_regressions.std(axis=1)
summary['r2'] = all_r2_scores.mean(axis=1)
summary.to_csv('../../data/summary.csv')

# Plot summary boxplot
keys = all_regressions.index

plot_keys = (key.replace('Stacked-multimodal', 'Stacked') for key in keys)
all_regressions.index = plot_keys

title = 'Age Prediction'
sns.set_style('whitegrid')
plt.figure()
ax = sns.boxplot(data=all_regressions.transpose(),
                 showmeans=True,
                 orient='h')
ax.set_title(title)
ax.set(xlabel='Mean Absolute Error (Years)')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black', bottom=True)

name = f'combined_plot.{OUT_FTYPE}'
plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')
