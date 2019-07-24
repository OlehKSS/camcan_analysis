"""Permutation importance for estimators"""
# the code was taken from
# https://github.com/scikit-learn/scikit-learn/pull/13146/files
import numpy as np
from joblib import Parallel
from joblib import delayed

from sklearn.metrics import check_scoring
from sklearn.utils import (Bunch, check_array,
                           check_random_state)


def _safe_column_setting(X, col_idx, values):
    """Set column on X using col_idx"""
    if hasattr(X, "iloc"):
        X.iloc[:, col_idx] = values
    else:
        X[:, col_idx] = values


def _safe_column_indexing(X, col_idx):
    """Return column from X using col_idx"""
    if hasattr(X, "iloc"):
        return X.iloc[:, col_idx].values
    else:
        return X[:, col_idx]


def _calculate_permutation_scores(estimator, X, y, col_idx, random_state,
                                  n_repeats, scorer):
    """Calculate score when ``col_idx`` is permuted."""
    original_feature = _safe_column_indexing(X, col_idx).copy()
    temp = original_feature.copy()

    scores = np.zeros(n_repeats)
    for n_round in range(n_repeats):
        random_state.shuffle(temp)
        _safe_column_setting(X, col_idx, temp)
        feature_score = scorer(estimator, X, y)
        scores[n_round] = feature_score

    _safe_column_setting(X, col_idx, original_feature)
    return scores


def permutation_importance(estimator, X, y, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=None):
    """Permutation importance for feature evaluation. [BRE]_

    The ``estimator`` is required to be a fitted estimator. ``X`` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by ``scoring``, is evaluated on a (potentially different) dataset
    defined by the ``X``. Next, a feature column from the validation set is
    permuted and the metric is evaluated again. The permutation importance is
    defined to be the difference between the baseline metric and metric from
    permutating the feature column.

    Read more in the :ref:`User Guide <permutation_importance>`.

    Parameters
    ----------
    estimator : object
        An estimator that has already been `fit` and is compatible with
        ``scorer``.

    X : numpy array or DataFrame, shape = (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape = (n_samples, ...)
        Targets for supervised or ``None`` for unsupervised.

    scoring : string, callable or None, optional (default=None)
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.

    n_repeats : int, optional (default=5)
        Number of times to permute a feature.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default None
        Pseudo-random number generator to control the permutations.
        See :term:`random_state`.

    Returns
    -------
    result : Bunch
        Dictionary-like object, with attributes:

        importances_mean : array, shape (n_features)
            Mean of feature importance over ``n_repeats``
        importances_std : array, shape (n_features)
            Standard deviation over ``n_repeats``
        importances : array, shape (n_features, n_repeats)
            Raw permutation importance scores.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
        2001.https://doi.org/10.1023/A:1010933404324

    """
    if hasattr(X, "iloc"):
        X = X.copy()  # Dataframe
    else:
        X = check_array(X, force_all_finite='allow-nan', dtype=np.object,
                        copy=True)

    random_state = check_random_state(random_state)
    scorer = check_scoring(estimator, scoring=scoring)

    baseline_score = scorer(estimator, X, y)
    scores = np.zeros((X.shape[1], n_repeats))

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
        estimator, X, y, col_idx, random_state, n_repeats, scorer
    ) for col_idx in range(X.shape[1]))

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances)
