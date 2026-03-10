"""
Helpers for plotting training vs validation behaviour (learning curves and
train/val development over iterations). Used in the Session 2 notebook to
teach overfitting, underfitting, and early stopping.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error


def plot_learning_curve(
    estimator,
    X,
    y,
    *,
    cv=5,
    train_sizes=None,
    scoring="neg_root_mean_squared_error",
    ax=None,
    title="Learning curve (train vs validation)",
):
    """
    Plot learning curve: mean train and validation score vs training set size.

    Uses sklearn's learning_curve. For RMSE we use neg_root_mean_squared_error;
    the plot shows positive RMSE (converted from negative scores).

    Parameters
    ----------
    estimator : sklearn estimator
    X, y : feature matrix and target
    cv : int or CV splitter
    train_sizes : array-like, optional
        Relative or absolute numbers of training examples. Default: np.linspace(0.1, 1.0, 10).
    scoring : str
        Sklearn scorer name. Default neg_root_mean_squared_error (we convert to positive for plot).
    ax : matplotlib axes, optional
    title : str

    Returns
    -------
    ax
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )
    # sklearn returns negative RMSE; convert to positive for display
    sign = -1 if scoring.startswith("neg_") else 1
    train_scores = sign * train_scores
    test_scores = sign * test_scores
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2
    )
    ax.fill_between(
        train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2
    )
    ax.plot(train_sizes_abs, train_mean, "o-", label="Train (RMSE)")
    ax.plot(train_sizes_abs, test_mean, "o-", label="Validation (RMSE)")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def compute_gbr_staged_rmse(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators=200,
    random_state=42,
    **kwargs,
):
    """
    Fit a GradientBoostingRegressor with n_estimators and return train/val RMSE
    at each boosting stage (1, 2, ..., n_estimators).

    Uses staged_predict for efficiency (single fit, then iterate predictions).

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    n_estimators : int
    random_state : int
    **kwargs : passed to GradientBoostingRegressor (e.g. max_depth, learning_rate).

    Returns
    -------
    steps : np.ndarray of shape (n_estimators,), 1..n_estimators
    train_rmse : np.ndarray of shape (n_estimators,)
    val_rmse : np.ndarray of shape (n_estimators,)
    model : fitted GradientBoostingRegressor (with n_estimators trees)
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, random_state=random_state, **kwargs
    )
    model.fit(X_train, y_train)

    train_rmse = []
    for y_pred in model.staged_predict(X_train):
        train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred)))
    val_rmse = []
    for y_pred in model.staged_predict(X_val):
        val_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    steps = np.arange(1, n_estimators + 1, dtype=int)
    return steps, np.array(train_rmse), np.array(val_rmse), model


def early_stop_index(val_scores, patience=0):
    """
    Index at which validation score is best (for early stopping).

    For RMSE/loss, "best" is minimum. If patience > 0, returns the first index
    after which the validation score did not improve for `patience` consecutive
    steps (optional more realistic early stopping).

    Parameters
    ----------
    val_scores : array-like, e.g. validation RMSE per step
    patience : int
        If > 0: stop when no improvement for this many steps. If 0: simply argmin.

    Returns
    -------
    int : index (0-based) at which to stop
    """
    val_scores = np.asarray(val_scores)
    best_idx = 0
    best_val = val_scores[0]
    for i in range(1, len(val_scores)):
        if val_scores[i] < best_val:
            best_val = val_scores[i]
            best_idx = i
        if patience > 0 and (i - best_idx) >= patience:
            return best_idx
    return best_idx


def plot_train_val_development(
    steps,
    train_scores,
    val_scores,
    early_stop_idx=None,
    *,
    xlabel="Number of boosting iterations",
    ylabel="RMSE",
    title="Train vs validation error development",
    ax=None,
):
    """
    Plot train and validation score (e.g. RMSE) vs step (e.g. n_estimators).
    Optionally mark the early-stopping step.

    Parameters
    ----------
    steps : array-like, e.g. [1, 2, ..., 200]
    train_scores, val_scores : array-like, same length as steps
    early_stop_idx : int or None
        Index (0-based) at which early stopping would occur; draw a vertical line.
    xlabel, ylabel, title : str
    ax : matplotlib axes, optional

    Returns
    -------
    ax
    """
    steps = np.asarray(steps)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_scores, label="Train", color="C0")
    ax.plot(steps, val_scores, label="Validation", color="C1")
    if early_stop_idx is not None and 0 <= early_stop_idx < len(steps):
        ax.axvline(
            steps[early_stop_idx],
            color="gray",
            linestyle="--",
            alpha=0.8,
            label=f"Early stop (step {steps[early_stop_idx]})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
