"""
Prepare boxplot of mean absolute error (MAE) for all modalities.

The information used for plotting will be output as a csv file.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIG_OUT_PATH = '../../data/figures/'
SCORES = '../../data/age_prediction_scores.csv'
OUT_FTYPE = 'pdf'

out_folder = os.path.join(FIG_OUT_PATH, 'boxplots')

all_regressions = pd.read_csv(SCORES).T

key_labels = {'Cortical Surface Area': 'Cortical Surface Area',
              'Cortical Thickness': 'Cortical Thickness',
              'Subcortical Volumes': 'Subcortical Volumes',
              # 'Connectivity Matrix, BASC 197 tan': 'BASC 197 tan',
              'Connectivity Matrix, MODL 256 tan': 'fMRI',
              # 'Connectivity Matrix, MODL 256 r2z': 'MODL 256 r2z',
              # 'Connectivity Matrix, BASC 197 r2z': 'BASC 197 r2z',
              # 'MEG': 'MEG',
              'MEG, Cortical Surface Area Stacked-multimodal': 'MEG + CSA',
              'MEG, Cortical Thickness Stacked-multimodal': 'MEG + CT',
              'MEG, Subcortical Volumes Stacked-multimodal': 'MEG + SV',
              # 'MEG, BASC 197 tan Stacked-multimodal': 'MEG, BASC 197 tan Stacked',
              'MEG, MODL 256 r2z Stacked-multimodal': 'MEG + fMRI',
              # 'MRI Stacked': 'MRI',
              # 'fMRI Stacked': 'fMRI',
              # 'MRI, fMRI Stacked-multimodal': 'MRI + fMRI',
              # 'MEG, MRI Stacked-multimodal': 'MEG + MRI',
              # 'MEG, fMRI Stacked-multimodal': 'MEG + fMRI',
              # 'MEG, MRI, fMRI Stacked-multimodal': 'MEG + MRI + fMRI'
              }


all_regressions = all_regressions.T[list(key_labels.keys())].T
all_regressions.index = key_labels.values()
all_regressions = all_regressions.iloc[::-1]

contrast = pd.DataFrame(all_regressions[4:].values - all_regressions[:4].values,
                        index=all_regressions[:4].index,
                        columns=all_regressions[:4].columns)

del all_regressions

colors = sns.color_palette("Blues", 3)
colors += [(1, 0, 0)]
# colors += sns.color_palette("Oranges", 1)
# colors += list(plt.cm.gray(np.linspace(0, 1, 4)))

fig, ax = plt.subplots()
bplot = plt.boxplot(contrast.values.T, vert=False, patch_artist=True, labels=contrast.index)

title = 'Age Prediction Gain'

# fill with colors
for patch, median, color in zip(bplot['boxes'], bplot['medians'], colors[::-1]):
    patch.set_facecolor(color)
    median.set_color("yellow")
ax.set_title(title)
ax.set(xlabel='Gain Mean Absolute Error (Years)')
fig.tight_layout()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black', bottom=True)
plt.show()

