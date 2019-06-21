"""Utilities for Jupyter Notebook reports."""
from itertools import combinations
from os import path

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (cross_val_score,
                                     cross_val_predict,
                                     learning_curve,
                                     ShuffleSplit,
                                     KFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..processing import StackingRegressor, SPoC


def run_meg_ridge(data, subjects_data, cv=10, alphas=None,
                  train_sizes=None, fbands=None):
    """Run ridge resgression on MEG data.

    Parameters
    ----------
    data : pandas.DataFrame
        Features to be used for predictions.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested using RidgeCV. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    fbands : [(float, float)]
        List of frequency bands to be checked with SPoC.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    # read sample data to prepare picks for epochs
    data_path = sample.data_path()
    raw_fname = path.join(data_path,
                          'MEG/sample/sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    picks = mne.pick_types(info, meg='mag')

    subjects = [d['subject'] for d in data if 'subject' in d]
    covs = np.array(tuple(d['covs'][:, picks][:, :, picks] for d
                          in data if 'subject' in d))

    # prepare data, subjects age
    y = subjects_data.loc[subjects].age.values
    X = np.arange(len(y))

    spoc = SPoC(covs=covs, fbands=fbands, spoc=True,
                n_components=len(picks), alpha=0.01)

    reg = make_pipeline(spoc, StandardScaler(),
                        RidgeCV(alphas))
    # Monte Carlo cross-validation
    cv_ss = ShuffleSplit(n_splits=cv, random_state=42)

    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=cv_ss)
    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv_ss)
    y_pred = cross_val_predict(reg, X, y, cv=cv)

    train_sizes, train_scores, test_scores =\
        learning_curve(reg, X, y, cv=cv_ss, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error')

    df_pred = pd.DataFrame(y_pred, index=subjects, dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def run_stacking(named_data, subjects_data, cv=10, alphas=None,
                 train_sizes=None, fbands=None, n_jobs=None):
    """Run stacking.

    Parameters
    ----------
    named_data : list(tuple(str, pandas.DataFrame))
        List of tuples (name, data) with name and corresponding features
        to be used for predictions by linear models.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    fbands : [(float, float)]
        List of frequency bands to be checked with SPoC.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    rnd_state = 42
    names = []
    combined_data = []
    meg_data = None
    # extract data and estimator names
    for name, data in named_data:
        names.append(name)
        if name == 'meg':
            meg_data = data
            meg_subjects = tuple(d['subject'] for d in data if 'subject' in d)
            pseudo_data = np.arange(len(meg_subjects))
            combined_data.append(pd.DataFrame(pseudo_data, index=meg_subjects))
        else:
            combined_data.append(data)

    data = pd.concat(combined_data, axis=1, join='inner')
    # if we have meg data, we will provide only one column of
    # data for the classifiers
    feature_col_lens = tuple(d.shape[1] for d in combined_data)
    estimators = []
    subjects = data.index.values
    # prepare first-level estimators for stacking
    for i_data, _ in enumerate(named_data):
        feature_transformers = []
        ft_begin = 0
        ft_end = 0
        # prepare input information for ColumnTransformer
        for i_ct, (name, col_len) in enumerate(zip(names, feature_col_lens)):
            trans_name = ('pass_' if i_data == i_ct else 'drop_') + name
            transformer = 'passthrough' if i_data == i_ct else 'drop'
            ft_end = ft_end + col_len
            trans_slice = slice(ft_begin, ft_end)
            ft_begin = ft_begin + col_len
            feature_transformers.append((trans_name, transformer, trans_slice))

        est_name = 'reg_' + named_data[i_data][0]

        if est_name == 'reg_meg':
            if fbands is None:
                raise ValueError('fbands should be given for MEG classifier.')
            # read sample data to prepare picks for epochs
            data_path = sample.data_path()
            raw_fname = path.join(data_path,
                                  'MEG/sample',
                                  'sample_audvis_filt-0-40_raw.fif')
            raw = mne.io.read_raw_fif(raw_fname)
            info = raw.info
            picks = mne.pick_types(info, meg='mag')
            # if there is no subject information than we'll skip that entry
            covs = np.array(tuple(d['covs'][:, picks][:, :, picks] for d
                                  in meg_data if 'subject' in d))
            spoc = SPoC(covs=covs, fbands=fbands, spoc=True,
                        n_components=len(picks), alpha=0.01)

            est_pipeline = make_pipeline(
                ColumnTransformer(feature_transformers),
                spoc, StandardScaler(), RidgeCV(alphas))
        else:
            est_pipeline = make_pipeline(
                ColumnTransformer(feature_transformers),
                StandardScaler(), RidgeCV(alphas))
        estimators.append((est_name, est_pipeline))

    final_estimator = RandomForestRegressor(n_estimators=100,
                                            random_state=rnd_state,
                                            oob_score=True, n_jobs=n_jobs)
    reg = StackingRegressor(estimators=estimators,
                            final_estimator=final_estimator, cv=cv,
                            random_state=rnd_state, n_jobs=n_jobs)

    y = subjects_data.loc[subjects].age.values
    X = data.values

    kfold_cv = KFold(n_splits=cv, shuffle=True, random_state=rnd_state)
    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=kfold_cv, n_jobs=n_jobs)

    r2 = cross_val_score(reg, X, y, scoring='r2', cv=kfold_cv, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=kfold_cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=kfold_cv, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)

    df_pred = pd.DataFrame(y_pred, index=subjects, dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def run_ridge(data, subjects_data, cv=10, alphas=None, train_sizes=None,
              n_jobs=None):
    """Run ridge resgression.

    Parameters
    ----------
    data : pandas.DataFrame
        Features to be used for predictions.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    # prepare data, subjects age
    subjects = data.index.values
    y = subjects_data.loc[subjects].age.values
    X = data.values

    reg = make_pipeline(StandardScaler(), RidgeCV(alphas))
    # Monte Carlo cross-validation
    cv_ss = ShuffleSplit(n_splits=cv, random_state=42)

    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=cv_ss, n_jobs=n_jobs)
    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv_ss, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=cv_ss, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error',  n_jobs=n_jobs)

    df_pred = pd.DataFrame(y_pred, index=subjects, dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def plot_pred(y, y_pred, mae, title='Prediction vs Measured'):
    """Plot predicted values vs real values."""
    plt.figure()
    plt.title(title)
    plt.scatter(y, y_pred,  edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=3, color='green')
    plt.plot([y.min(), y.max()], [y.min() - mae, y.max() - mae], 'k--', lw=3,
             color='red')
    plt.plot([y.min(), y.max()], [y.min() + mae, y.max() + mae], 'k--', lw=3,
             color='red')
    plt.xlabel('Chronological Age')
    plt.ylabel('Predicted Age')
    plt.grid()
    plt.show()


# https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_learning_curve.html
def plot_learning_curve(train_sizes, train_scores, test_scores,
                        title='Learning Curves', ylim=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Numbers of training examples that has been used to generate
        the learning curve.

    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    """
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
    plt.show()


def plot_barchart(mae_std,
                  title='Age Prediction Performance of Different Modalities',
                  bar_text_indent=2):
    """Plot bar chart.

    Parameters
    ----------
    mae_std : dict(str, (number, number))
        Dictionary with labels and corresponding mae and std.
    title : str
        Bar chart title.
    bar_text_indent : number
        Indent from the bar top for labels displaying mae and std,
        measured in years.

    """
    objects = tuple(reversed(sorted(mae_std.keys())))
    y_pos = np.arange(len(objects))
    mae = tuple(mae_std[k][0] for k in objects)
    std = tuple(mae_std[k][1] for k in objects)

    fig, axs = plt.subplots()
    axs.barh(y_pos, mae, align='center', xerr=std)

    # remove frame around the plot
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)

    for i, v in enumerate(mae):
        axs.text(v + bar_text_indent, i - 0.05,
                 f'{round(v, 2)} ({round(std[i], 2)})',
                 color='blue', bbox=dict(facecolor='white'))

    plt.yticks(y_pos, objects)
    plt.xlabel('Absolute Prediction Error (Years)')
    plt.title(title)
    plt.show()


def plot_boxplot(data, title='Age Prediction Performance'):
    """Plot box plot.

    Parameters
    ----------
    data : dict(str, numpy.ndarray)
        Dictionary with labels and corresponding data.
    title : str
        Bar chart title.

    """
    data_pd = pd.DataFrame(data)
    sns.set_style('darkgrid')
    plt.figure()
    ax = sns.boxplot(data=data_pd, showmeans=True, orient='h')
    ax.set_title(title)
    ax.set(xlabel='Absolute Prediction Error (Years)')
    plt.show()


def plot_error_scatters(data, title='AE Scatter', xlim=None, ylim=None):
    """Plot prediction errors of different modalities versus each other."""
    data = data.dropna()
    age = data.age.values
    color_map = plt.cm.viridis((age - min(age)) / max(age))
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    for key1, key2 in combinations(keys, r=2):
        fig, ax = plt.subplots()
        x_values = np.abs(data[key1].values - age)
        y_values = np.abs(data[key2].values - age)
        plt.scatter(x_values, y_values, edgecolors='black', color=color_map)
        plt.title(title)
        plt.xlabel(key1)
        plt.ylabel(key2)

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


def plot_error_age(data, title='AE vs Age', xlim=None, ylim=None):
    """Plot prediction errors of different modalities versus subject's age."""
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    for key1 in keys:
        data_slice = data[key1].dropna()
        age = data.loc[data_slice.index, 'age'].values
        abs_errors = np.abs(data_slice.values - age)
        plt.figure()
        plt.scatter(age, abs_errors, edgecolors='black')
        plt.title(title)
        plt.xlabel('Age (Years)')
        plt.ylabel(key1)
        plt.grid()

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)


def plot_error_segments(data, segment_len=10, title=None, figsize=None,
                        xlim=(0, 55)):
    """Plot prediction errors for different age groups."""
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    age = data.age.values
    for key in keys:
        n_segments = int((age.max() - age.min()) / segment_len)
        segments_dict = {}
        plt_title = 'AE per Segment, %s' % key if title is None else title
        age_pred = data[key]

        for i in range(0, n_segments):
            bound_low = age.min() + i * segment_len
            bound_high = age.min() + (i + 1) * segment_len

            if i == n_segments - 1:
                indices = age >= bound_low
            else:
                indices = (age >= bound_low) * (age < bound_high)

            segments_dict[f'{bound_low}-{bound_high}'] =\
                np.abs(age[indices] - age_pred[indices])

        df = pd.DataFrame.from_dict(segments_dict, orient='index').transpose()

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df, showmeans=True, orient='h')
        ax.set_title(plt_title)
        ax.set(xlim=xlim, xlabel='Absolute Prediction Error (Years)',
               ylabel='Age Ranges')
        plt.show()
