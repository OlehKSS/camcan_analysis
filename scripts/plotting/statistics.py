"""Prepare plots of statistical significance of features."""
import os

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import seaborn as sns
from sklearn.inspection import partial_dependence

from camcan.processing import permutation_importance
from camcan.utils import train_stacked_regressor

# common subjects 574
CV = 10
N_JOBS = 4
PANDAS_OUT_FILE = '../../data/age_prediction_exp_data.h5'
STRUCTURAL_DATA = '../../data/structural/structural_data.h5'
CONNECT_DATA_CORR = '../../data/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = '../../data/connectivity/connect_data_tangent.h5'
MEG_SOURCE_SPACE_DATA = '../../data/meg_source_space_data.h5'
FREQ_BANDS = ('alpha',
              'beta_high',
              'beta_low',
              'delta',
              'gamma_high',
              'gamma_lo',
              'gamma_mid',
              'low',
              'theta')
# store mae, learning curves for summary plots
regression_mae = pd.DataFrame(columns=range(0, CV), dtype=float)
regression_r2 = pd.DataFrame(columns=range(0, CV), dtype=float)
learning_curves = {}

# read information about subjects
subjects_data = pd.read_csv('../../data/participant_data.csv', index_col=0)
# for storing predictors data
subjects_predictions = pd.DataFrame(subjects_data.age,
                                    index=subjects_data.index,
                                    dtype=float)

# 595 subjects
meg_data = pd.read_hdf(MEG_SOURCE_SPACE_DATA, key='meg')
meg_subjects = set(meg_data['subject'])

# read features
area_data = pd.read_hdf(STRUCTURAL_DATA, key='area')
thickness_data = pd.read_hdf(STRUCTURAL_DATA, key='thickness')
volume_data = pd.read_hdf(STRUCTURAL_DATA, key='volume')

area_data = area_data.dropna()
thickness_data = thickness_data.dropna()
volume_data = volume_data.dropna()

# take only subjects that are both in MEG and Structural MRI
structural_subjects = set(area_data.index)
common_subjects = meg_subjects.intersection(structural_subjects)

area_data = area_data.loc[common_subjects]
thickness_data = thickness_data.loc[common_subjects]
volume_data = volume_data.loc[common_subjects]
meg_data = meg_data[meg_data.subject.isin(common_subjects)]

# read connectivity data
connect_data_tangent_basc = pd.read_hdf(CONNECT_DATA_TAN, key='basc197')
connect_data_r2z_basc = pd.read_hdf(CONNECT_DATA_CORR, key='basc197')
connect_data_tangent_modl = pd.read_hdf(CONNECT_DATA_TAN, key='modl256')
connect_data_r2z_modl = pd.read_hdf(CONNECT_DATA_CORR, key='modl256')

# use only common subjects
connect_data_tangent_basc = connect_data_tangent_basc.loc[common_subjects]
connect_data_r2z_basc = connect_data_r2z_basc.loc[common_subjects]
connect_data_tangent_modl = connect_data_tangent_modl.loc[common_subjects]
connect_data_r2z_modl = connect_data_r2z_modl.loc[common_subjects]

print('Data was read successfully.')

key = 'MEG, MRI, fMRI Stacked-multimodal'
data = [('area', area_data),
        ('thickness', thickness_data),
        ('volume', volume_data),
        ('basc', connect_data_tangent_basc),
        ('meg', meg_data)]

reg, X, y = train_stacked_regressor(data, subjects_data, cv=CV,
                                    fbands=FREQ_BANDS)

# plot feature importance from GINI
FIG_OUT_PATH = '../../data/figures/'
OUT_FTYPE = 'pdf'

out_folder = os.path.join(FIG_OUT_PATH, 'boxplots')

title = 'Feature Importance'

plot_keys = ('CSA',  # Cortical Surface Area
             'CT',  # Cortical Thickness
             'SV',  # Subcortical Volumes
             'fMRI',
             'MEG')

colors = sns.color_palette("Blues", 3)
colors += [(1, 0, 0)]
colors += sns.color_palette("Oranges", 1)

plot_data = pd.DataFrame(reg.final_estimator_.feature_importances_,
                         index=plot_keys)
plot_data['colors'] = colors
plot_data = plot_data.sort_values(0, ascending=False)

sns.set(style='whitegrid')
plt.figure()
ax = sns.barplot(x=plot_data[0], y=plot_data.index, palette=plot_data.colors)
ax.set_title(title)
ax.set(xlabel='Importance')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black', bottom=True)

name = f'feature_importance_rf.{OUT_FTYPE}'
plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')

# plot partial dependence
title = 'Partial Dependence'
features = [0, 1, 2, 3, 4]

pd_result = Parallel(n_jobs=len(features))(
    delayed(partial_dependence)(reg.final_estimator_,
                                reg.transform(X), f)
    for f in features)

fig = plt.figure()
style.use('default')
plt.grid()

for index, (pdp, axes) in enumerate(pd_result):
    plt.plot(axes[0], pdp[0, :], color=colors[index], label=plot_keys[index])

plt.title(title)
plt.legend()

name = f'partial_dependence.{OUT_FTYPE}'
plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')

# plot permutation importance
title = 'Permutation Importances'
result = permutation_importance(reg.final_estimator_, reg.transform(X), y,
                                n_repeats=10, random_state=42)

plot_data = pd.DataFrame(result.importances_mean,
                         index=plot_keys)
plot_data['colors'] = colors
plot_data = plot_data.sort_values(0, ascending=False)

sns.set(style='whitegrid')
plt.figure()
ax = sns.barplot(x=plot_data[0], y=plot_data.index, palette=plot_data.colors)
ax.set_title(title)
ax.set(xlabel='Importance')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black', bottom=True)

name = f'permutation_importance.{OUT_FTYPE}'
plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')
