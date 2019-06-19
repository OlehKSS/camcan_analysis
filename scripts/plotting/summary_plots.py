import os
import pickle as pkl
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_OUT_PATH = '../../data/figures/'
PREDICTIONS = '../../data/age_prediction_exp_data.h5'
OUT_FTYPE = 'pdf'

out_folder = os.path.join(FIG_OUT_PATH, 'boxplots')

all_regressions = pd.read_hdf(PREDICTIONS, key='regression')
all_r2_scores = pd.read_hdf(PREDICTIONS, key='r2')

key_labels = {'Cortical Surface Area': 'Cortical Surface Area',
            'Cortical Thickness': 'Cortical Thickness',
            'Subcortical Volumes': 'Subcortical Volumes',
            'Connectivity Matrix, BASC 197 tan': 'BASC 197 tan',
            'Connectivity Matrix, MODL 256 tan': 'MODL 256 tan',
            'Connectivity Matrix, MODL 256 r2z': 'MODL 256 r2z',
            'Connectivity Matrix, BASC 197 r2z': 'BASC 197 r2z',
            'MEG': 'MEG',
            'MEG, Cortical Surface Area Stacked-multimodal': 'MEG, CSA Stacked',
            'MEG, Cortical Thickness Stacked-multimodal': 'MEG, CT Stacked',
            'MEG, Subcortical Volumes Stacked-multimodal': 'MEG, SV Stacked',
            'MEG, BASC 197 tan Stacked-multimodal': 'MEG, BASC 197 tan Stacked',
            'MEG, MODL 256 r2z Stacked-multimodal': 'MEG, MODL 256 r2z Stacked',
            'MRI Stacked': 'MRI Stacked',
            'fMRI Stacked': 'fMRI Stacked',
            'MRI, fMRI Stacked-multimodal': 'MRI, fMRI Stacked',
            'MEG, MRI Stacked-multimodal': 'MEG, MRI Stacked',
            'MEG, fMRI Stacked-multimodal': 'MEG, fMRI Stacked',
            'MEG, MRI, fMRI Stacked-multimodal': 'MEG, MRI, fMRI Stacked'}

out_keys = (key_labels[key] for key in all_regressions.index)
# save mae, std to a csv file
summary = pd.DataFrame(index=all_regressions.index,
                       columns=['mae', 'std'], dtype=float)
summary['mae'] = all_regressions.mean(axis=1)
summary['std'] = all_regressions.std(axis=1)
summary['r2'] = all_r2_scores.mean(axis=1)
summary.to_csv('../../data/summary.csv')

all_regressions.index = out_keys

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
