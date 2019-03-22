"""Utilities for Jupyter Notebook reports"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bagging import CVBagging
from .stacking import StackingRegressor


def run_stacking(named_data, subjects_data, cv=10, alphas=None, train_sizes=None, n_jobs=None):
    """Helper for running ridge resgression.
    
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
        Values for parameter alpha to be tested using. Default is
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
        alphas = np.logspace(start=-3, stop=1, num=50, base=10.0)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    rnd_state = 42
    names = tuple(n for n, _ in named_data)
    data =  pd.concat((d for _, d in named_data), axis=1, join='inner')
    feature_col_lens = tuple(d.shape[1] for _, d in named_data)
    estimators = []
    # prepare first-level estimators for stacking
    for i_data, _ in enumerate(named_data):
        feature_transformers = []
        ft_begin = 0
        ft_end = 0
        # prepare input information for ColumnTransformer
        for i_ct, (name, col_len) in  enumerate(zip(names, feature_col_lens)):
            trans_name = ('pass_' if i_data == i_ct else 'drop_') + name
            transformer = 'passthrough' if i_data == i_ct else 'drop'
            ft_end = ft_end + col_len
            trans_slice = slice(ft_begin, ft_end)
            ft_begin = ft_begin + col_len
            feature_transformers.append((trans_name, transformer, trans_slice))

        est_name = 'reg_' + named_data[i_data][0]
        est_pipeline = make_pipeline(ColumnTransformer(feature_transformers), 
                                                       StandardScaler(),
                                                       CVBagging(alphas))
        estimators.append((est_name, est_pipeline))

    final_estimator = RandomForestRegressor(n_estimators=100, random_state=rnd_state,
                                            oob_score=True, n_jobs=n_jobs)
    reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=cv,
                            random_state=rnd_state, n_jobs=n_jobs)

    data_rnd = data.sample(frac=1)
    subjects = data_rnd.index.values
    y = subjects_data.loc[data_rnd.index.values].age.values
    X = data_rnd.values
    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=n_jobs)

    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=cv, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)

    return y, y_pred, mae, r2, train_sizes, train_scores, test_scores, subjects


def run_ridge(data, subjects_data, cv=10, alphas=None, train_sizes=None, n_jobs=None):
    """Helper for running ridge resgression.
    
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
        Values for parameter alpha to be tested using. Default is
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
        alphas = np.logspace(start=-3, stop=1, num=50, base=10.0)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)
    
    # prepare data, subjects age
    data_rnd = data.sample(frac=1)
    subjects = data_rnd.index.values
    y = subjects_data.loc[data_rnd.index.values].age.values
    X = data_rnd.values

    reg = make_pipeline(StandardScaler(), CVBagging(alphas))
    # Monte Carlo cross-validation
    cv_ss = ShuffleSplit(n_splits=cv, random_state=42)

    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=cv_ss, n_jobs=n_jobs)
    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv_ss, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)
    
    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=cv_ss, train_sizes=train_sizes, scoring='neg_mean_absolute_error',  n_jobs=n_jobs)

    return y, y_pred, mae, r2, train_sizes, train_scores, test_scores, subjects


def plot_pred(y, y_pred, mae, title='Prediction vs Measured'):
    """Plot predicted values vs real values."""
    plt.figure()
    plt.title(title)
    plt.scatter(y, y_pred,  edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=3, color='green')
    plt.plot([y.min(), y.max()], [y.min() - mae, y.max() - mae], 'k--', lw=3, color='red')
    plt.plot([y.min(), y.max()], [y.min() + mae, y.max() + mae], 'k--', lw=3, color='red')
    plt.xlabel('chronological age')
    plt.ylabel('predicted age')
    plt.grid()
    plt.show()

    
# https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning Curves', ylim=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------        
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Numbers of training examples that has been used to generate the learning curve.
        
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


def plot_barchart(mae_std, title='Age Prediction Performance of Different Modalities',
                  bar_text_indent=2):
    """Plot bar chart.
    
    Parameters
    ----------
    mae_std : dict(str, (number, number))
        Dictionary with labels and corresponding mae and std.
    title : str
        Bar chart title.
    bar_text_indent : number
        Indent from the bar top for labels displaying mae and std, measures in years.
    """
    objects = tuple(reversed(sorted(mae_std.keys())))
    y_pos = np.arange(len(objects))
    mae = tuple(mae_std[k][0] for k in objects)
    std = tuple(mae_std[k][1] for k in objects)

    fig, axs = plt.subplots()
    #axs.grid(zorder=0)
    axs.barh(y_pos, mae, align='center', xerr=std)

    # remove frame around the plot
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)

    for i, v in enumerate(mae):
        axs.text(v + bar_text_indent, i - 0.05, f'{round(v, 2)} ({round(std[i], 2)})', 
                 color='blue', bbox=dict(facecolor='white'))

    plt.yticks(y_pos, objects)
    plt.xlabel('Absolute Prediction Error (Years)')
    plt.title(title)
    plt.show()

