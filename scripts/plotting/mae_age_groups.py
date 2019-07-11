"""Plot mean absolute error (MAE) versus age group."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'pdf'

data = pd.read_hdf(PREDICTIONS, key='predictions')
data = data.dropna()

colors = sns.color_palette("Blues", 3)
colors += [(1, 0, 0)]
colors += sns.color_palette("Oranges", 1)
colors += list(reversed(sns.light_palette('black', 4)))

# Plot summary boxplot
keys = ['Cortical Surface Area',
        'Cortical Thickness',
        'Subcortical Volumes',
        'Connectivity Matrix, BASC 197 tan',
        'MEG',
        'MRI, fMRI Stacked-multimodal',
        'MEG, MRI Stacked-multimodal',
        'MEG, fMRI Stacked-multimodal',
        'MEG, MRI, fMRI Stacked-multimodal']

plt_labels = {'Cortical Surface Area': 'Cortical Surface Area',
              'Cortical Thickness': 'Cortical Thickness',
              'Subcortical Volumes': 'Subcortical Volumes',
              'Connectivity Matrix, BASC 197 tan': 'fMRI',
              'MEG': 'MEG',
              'MRI, fMRI Stacked-multimodal': 'MRI + fMRI',
              'MEG, MRI Stacked-multimodal': 'MEG + MRI',
              'MEG, fMRI Stacked-multimodal': 'MEG + fMRI',
              'MEG, MRI, fMRI Stacked-multimodal': 'MEG + MRI + fMRI'}

plt_colors = {k: c for k, c in zip(keys, colors)}

age = data.age.values

title = 'MAE per Age Group'
segment_len = 10  # years

fig, ax = plt.subplots()
for key in keys:
    x_ticks = []
    x_labels = []
    y_key = []
    n_segments = int((age.max() - age.min()) / segment_len)
    age_pred = data[key]

    for i in range(0, n_segments):
        bound_low = age.min() + i * segment_len
        bound_high = age.min() + (i + 1) * segment_len

        if i == n_segments - 1:
            indices = age >= bound_low
        else:
            indices = (age >= bound_low) * (age < bound_high)

        y_key.append(np.abs(age[indices] - age_pred[indices]).values.mean())

        if x_ticks is not None:
            bound_mid = 0.5 * (bound_low + bound_high)
            x_ticks.append(bound_mid)
        if x_labels is not None:
            x_labels.append(f'{int(bound_low)}-{int(bound_high - 1)}')

    line_style = '--o' if 'Stacked' in key else '-o'
    plt.plot(x_ticks, y_key, line_style, color=plt_colors[key],
             label=plt_labels[key])

ax.set(xlim=(age.min(), age.max()), xlabel='Age Group (Years)',
       ylabel='MAE (Years)')
plt.xticks(ticks=x_ticks, labels=x_labels)
plt.title(title)
plt.grid()
plt.legend()

name = f'mae_age_group.{OUT_FTYPE}'
plt.savefig(os.path.join(FIG_OUT_PATH, name), bbox_inches='tight')
