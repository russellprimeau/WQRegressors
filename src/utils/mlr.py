"""Multiple Linear Regression with independent feature selection.

Feature selection pipeline (training data only):
  1. Drop constant / near-constant columns
  2. Mutual information — drop features below a quantile threshold
  3. L1 / Lasso (LassoCV) — retain non-zero coefficients
  4. Variance Inflation Factor — iteratively remove highest VIF > threshold

Usage:
    from utils.mlr import evaluate_mlr
    predictions, targets = evaluate_mlr(train_samples, test_samples)
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _drop_constant(X, feature_idx):
    """Remove features with zero variance."""
    keep = []
    for i, col_idx in enumerate(feature_idx):
        col = X[:, i]
        finite = col[np.isfinite(col)]
        if len(finite) >= 2 and np.std(finite, ddof=1) > 0:
            keep.append(col_idx)
    return keep


def _select_by_mutual_info(X, y, feature_idx, mi_quantile=0.25, random_state=0):
    """Drop features whose MI with y falls below the *mi_quantile* quantile."""
    if len(feature_idx) <= 1:
        return feature_idx

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 3:
        return feature_idx

    mi = mutual_info_regression(X[mask], y[mask], random_state=random_state)
    threshold = np.quantile(mi, mi_quantile)
    return [idx for idx, score in zip(feature_idx, mi) if score > threshold]


def _select_by_lasso(X, y, feature_idx, random_state=0):
    """Retain features with non-zero Lasso coefficients (CV-tuned alpha)."""
    if len(feature_idx) <= 1:
        return feature_idx

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 5:
        return feature_idx

    lasso = LassoCV(cv=min(5, mask.sum()), random_state=random_state, max_iter=20000, tol=2e-4)
    lasso.fit(X[mask], y[mask])
    nonzero = np.abs(lasso.coef_) > 0
    selected = [idx for idx, keep in zip(feature_idx, nonzero) if keep]
    return selected if selected else feature_idx  # fallback: keep all if Lasso zeroes everything


def _compute_vif(X):
    """Compute VIF for each column of X using numpy (no statsmodels dependency).

    VIF_j = 1 / (1 - R²_j), where R²_j is from regressing column j on all others.
    """
    n, p = X.shape
    vifs = np.full(p, np.inf)
    if p <= 1 or n < 3:
        return np.ones(p)
    for j in range(p):
        others = np.delete(X, j, axis=1)
        # Add intercept
        others_aug = np.column_stack([np.ones(n), others])
        coef, residuals, _, _ = np.linalg.lstsq(others_aug, X[:, j], rcond=None)
        y_pred = others_aug @ coef
        ss_res = np.sum((X[:, j] - y_pred) ** 2)
        ss_tot = np.sum((X[:, j] - np.mean(X[:, j])) ** 2)
        if ss_tot == 0:
            vifs[j] = np.inf
        else:
            r2 = 1.0 - ss_res / ss_tot
            vifs[j] = 1.0 / max(1.0 - r2, 1e-10)
    return vifs


def _select_by_vif(X, y, feature_idx, vif_threshold=10.0):
    """Iteratively remove the feature with highest VIF until all are below threshold."""
    if len(feature_idx) <= 1:
        return feature_idx

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 3:
        return feature_idx

    Xv = X[mask].copy()
    idx_list = list(range(len(feature_idx)))

    while len(idx_list) > 1:
        vifs = _compute_vif(Xv[:, idx_list])
        worst = int(np.argmax(vifs))
        if vifs[worst] <= vif_threshold:
            break
        idx_list.pop(worst)

    return [feature_idx[i] for i in idx_list]


def select_features(X_train, y_train, feature_names,
                    mi_quantile=0.25, vif_threshold=10.0, random_state=0):
    """Run the full feature selection pipeline on training data.

    Parameters
    ----------
    X_train : ndarray [n_samples, n_features]
    y_train : ndarray [n_samples]  (single target column)
    feature_names : list[str]
    mi_quantile : float — MI scores below this quantile are dropped
    vif_threshold : float — iteratively remove features with VIF above this
    random_state : int

    Returns
    -------
    selected_idx : list[int] — indices into original feature axis
    selected_names : list[str]
    """
    all_idx = list(range(X_train.shape[1]))

    # Step 1: drop constant
    idx = _drop_constant(X_train, all_idx)
    if not idx:
        return [], []

    # Step 2: mutual information
    X_sub = X_train[:, idx]
    idx = _select_by_mutual_info(X_sub, y_train, idx, mi_quantile, random_state)
    if not idx:
        return [], []

    # Step 3: L1 / Lasso
    X_sub = X_train[:, idx]
    idx = _select_by_lasso(X_sub, y_train, idx, random_state)
    if not idx:
        return [], []

    # Step 4: VIF
    X_sub = X_train[:, idx]
    idx = _select_by_vif(X_sub, y_train, idx, vif_threshold)

    names = [feature_names[i] for i in idx] if feature_names else []
    return idx, names


# ---------------------------------------------------------------------------
# Fit & predict
# ---------------------------------------------------------------------------

def fit_and_predict(X_train, y_train, X_test, feature_names=None, verbose=False):
    """Feature-select, fit MLR on training data, predict on test data.

    Parameters
    ----------
    X_train, X_test : ndarray [n, d]
    y_train : ndarray [n]  (single target)
    feature_names : list[str] or None

    Returns
    -------
    predictions : ndarray [m]  (NaN where test inputs have NaN)
    metadata : dict  (selected_features, n_train, coefficients, intercept)
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    sel_idx, sel_names = select_features(X_train, y_train, feature_names)

    predictions = np.full(X_test.shape[0], np.nan)
    meta = {"selected_features": sel_names, "n_selected": len(sel_idx),
            "n_train": 0, "coefficients": [], "intercept": np.nan}

    if not sel_idx:
        if verbose:
            print("[MLR] No features survived selection; returning NaN predictions.")
        return predictions, meta

    Xtr = X_train[:, sel_idx]
    Xte = X_test[:, sel_idx]

    train_mask = np.all(np.isfinite(Xtr), axis=1) & np.isfinite(y_train)
    test_mask = np.all(np.isfinite(Xte), axis=1)

    if train_mask.sum() < 2:
        if verbose:
            print("[MLR] Fewer than 2 valid training rows; returning NaN predictions.")
        return predictions, meta

    model = LinearRegression()
    model.fit(Xtr[train_mask], y_train[train_mask])
    predictions[test_mask] = model.predict(Xte[test_mask])

    meta["n_train"] = int(train_mask.sum())
    meta["coefficients"] = model.coef_.tolist()
    meta["intercept"] = float(model.intercept_)

    if verbose:
        print(f"[MLR] {len(sel_idx)} features selected: {sel_names}")
        print(f"[MLR] Trained on {train_mask.sum()} samples, predicting {test_mask.sum()} test samples.")

    return predictions, meta


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

def evaluate_mlr(train_samples, test_samples, feature_names=None, verbose=False):
    """Multiple Linear Regression with independent feature selection.

    Matches the interface pattern of other model evaluators.

    Parameters
    ----------
    train_samples : list of (X, y, filename) tuples
    test_samples : list of (X, y, filename) tuples
    feature_names : list[str] or None — names for each input feature column
    verbose : bool

    Returns
    -------
    predictions : ndarray [n_test, n_outputs]
    targets : ndarray [n_test, n_outputs]
    metadata : list[dict] — per-target metadata (selected features, coefficients, etc.)
    """
    X_train = np.array([s[0].flatten() for s in train_samples], dtype=float)
    y_train = np.array([s[1].flatten() for s in train_samples], dtype=float)
    X_test = np.array([s[0].flatten() for s in test_samples], dtype=float)
    y_test = np.array([s[1].flatten() for s in test_samples], dtype=float)

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Expand feature_names to match flattened dimension (n_rows * n_cols)
    n_flat = X_train.shape[1]
    if feature_names is not None and len(feature_names) < n_flat:
        n_rows_per_sample = n_flat // len(feature_names)
        feature_names = [
            f"{name}_r{r}" for r in range(n_rows_per_sample) for name in feature_names
        ]

    n_outputs = y_train.shape[1]
    predictions = np.full_like(y_test, np.nan)
    all_meta = []

    for j in range(n_outputs):
        pred_j, meta_j = fit_and_predict(
            X_train, y_train[:, j], X_test,
            feature_names=feature_names, verbose=verbose,
        )
        predictions[:, j] = pred_j
        all_meta.append(meta_j)

    return predictions, y_test, all_meta
