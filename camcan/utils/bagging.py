import numpy as np
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import indexable


class CVBagging(BaseEstimator, RegressorMixin):
    """Perform CV Bagging with ridge regression.

    Parameters
    ----------
    alphas : numpy array of shape [n_alphas]
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC.
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
    """
    def __init__(self, alphas=(0.1, 1.0, 10.0), cv=10):
        self.alphas = np.asarray(alphas)
        self.cv = cv
    
    def fit(self, X, y, n_jobs=None):
        X, y = indexable(X, y)
        cv = check_cv(self.cv, y)

        def call_ridgecv(X, y, train):
            return RidgeCV(alphas=self.alphas).fit(X[train], y[train])

        parallel = Parallel(n_jobs=n_jobs)
        self.estimators_ = parallel(delayed(call_ridgecv)(X, y, train)
            for train, _ in cv.split(X, y))
            
        return self
    
    def predict(self, X):
        predictions = []

        for est in self.estimators_:
            predictions.append(est.predict(X))
        
        return np.array(predictions).mean(axis=0)

