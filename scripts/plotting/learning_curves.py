import os
import pickle as pkl
import shutil

import matplotlib.pyplot as plt
import numpy as np


FIG_OUT_PATH = '../../data/figures/'
LEARNING_CURVES = '../../data/learning_curves.pkl'
OUT_FTYPE = 'png'

with open(LEARNING_CURVES, 'rb') as handle:
    data = pkl.load(handle)

ylim = None
out_folder = os.path.join(FIG_OUT_PATH, 'learning_curves')

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.mkdir(out_folder)
else:
    os.mkdir(out_folder)

for key in data:
    train_scores = data[key]['train_scores']
    train_sizes = data[key]['train_sizes']
    test_scores = data[key]['test_scores']

    title = f'Learning Curve, {key}'
    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_scores = -train_scores
    test_scores = -test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
                label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
                label='Cross-validation score')

    plt.legend(loc='best')
    name = f'LC_{key.replace(" ", "-")}.{OUT_FTYPE}'
    plt.savefig(os.path.join(out_folder, name), bbox_inches='tight')
