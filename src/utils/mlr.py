"""Multiple Linear Regression with independent feature selection.

Feature selection pipeline (training data only):
  0. Spearman pre-filter — on *base* predictor columns (averaged across time
     steps), keep only columns with significant correlation to the target
     (p < 0.05 and |ρ| > 0.20).  Reduces dimensionality before flattening.
  1. Drop constant / near-constant columns
  2. Mutual information — keep top features by MI score (above quantile)
  3. L1 / Lasso (LassoCV) — retain non-zero coefficients; cap at *max_lasso_features*
  4. Variance Inflation Factor — iteratively remove highest VIF > threshold

Computational guard: VIF is O(p²) per iteration, so features entering
VIF are capped at 50 (top by |Lasso coefficient|) with 50 max iterations.

Usage:
    from utils.mlr import evaluate_mlr
    predictions, targets = evaluate_mlr(train_samples, test_samples)
"""

import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression

# MI and Lasso are efficient (sklearn-optimised); only VIF is O(p²) per
# iteration with numpy lstsq, so that is the stage that needs a hard cap.
_MAX_VIF_FEATURES = 500     # max features entering VIF (top by |Lasso coef|)
_MAX_VIF_ITERATIONS = 500   # max removal rounds inside VIF


# ---------------------------------------------------------------------------
# Spearman pre-filter (operates on base columns before flattening)
# ---------------------------------------------------------------------------

def _prefilter_by_spearman(train_samples, target_idx, base_feature_names,
                           p_threshold=0.05, rho_threshold=0.20):
    """Pre-filter base predictor columns using Spearman's rank correlation.

    For each base column, compute the mean across time steps (rows) per
    training sample, then correlate that vector with the target across
    samples.  Keep columns where *both* p < *p_threshold* and |ρ| >
    *rho_threshold*.

    Parameters
    ----------
    train_samples : list of (X, y, filename) — X is (n_rows, n_cols)
    target_idx : int — index into the target vector for this output
    base_feature_names : list[str] — one name per input column
    p_threshold : float
    rho_threshold : float

    Returns
    -------
    keep_cols : list[int] — indices of base columns that pass the filter
    spearman_results : dict[str, (rho, p)] — per-column results for metadata
    """
    n_cols = len(base_feature_names)

    # Build per-sample column means and target values
    col_means = []
    targets = []
    for s in train_samples:
        X = np.asarray(s[0], dtype=float)   # (n_rows, n_cols)
        y = np.asarray(s[1], dtype=float).flatten()
        y_val = y[target_idx] if target_idx < len(y) else y[0]
        if not np.isfinite(y_val):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_means.append(np.nanmean(X, axis=0))  # (n_cols,)
        targets.append(y_val)

    spearman_results = {}

    if len(targets) < 5:
        # Not enough data — keep everything
        return list(range(n_cols)), spearman_results

    col_means = np.array(col_means)  # (n_valid, n_cols)
    targets = np.array(targets)

    keep_cols = []
    for c in range(n_cols):
        col = col_means[:, c]
        finite = np.isfinite(col) & np.isfinite(targets)
        if finite.sum() < 5:
            continue
        rho, p = spearmanr(col[finite], targets[finite])
        spearman_results[base_feature_names[c]] = (float(rho), float(p))
        if p < p_threshold and abs(rho) > rho_threshold:
            keep_cols.append(c)

    if not keep_cols:
        # Fallback: keep all if nothing passes (avoid empty selection)
        return list(range(n_cols)), spearman_results

    return keep_cols, spearman_results


def get_spearman_features(train_samples, feature_names, target_idx=0,
                          p_threshold=0.05, rho_threshold=0.20):
    """Return base column names that pass the Spearman pre-filter.

    Convenience wrapper around :func:`_prefilter_by_spearman` for use
    by the sweep pipeline (which needs the filtered columns without running
    the full MLR evaluation).

    Returns
    -------
    kept_names : list[str] — base column names that pass the filter
    spearman_results : dict[str, (rho, p)] — per-column correlation results
    """
    keep_cols, results = _prefilter_by_spearman(
        train_samples, target_idx, feature_names,
        p_threshold=p_threshold, rho_threshold=rho_threshold,
    )
    return [feature_names[c] for c in keep_cols], results


# ---------------------------------------------------------------------------
# Feature selection (on flattened features)
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
    """Retain features with non-zero Lasso coefficients (CV-tuned alpha).

    If more than _MAX_VIF_FEATURES survive, keep those with the largest
    absolute coefficients so VIF remains tractable.
    """
    if len(feature_idx) <= 1:
        return feature_idx

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 5:
        return feature_idx

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
        lasso = LassoCV(cv=min(5, mask.sum()), random_state=random_state,
                        max_iter=20000, tol=5e-4)
        lasso.fit(X[mask], y[mask])

    coef_abs = np.abs(lasso.coef_)
    nonzero_mask = coef_abs > 0
    selected = [(idx, c) for idx, keep, c in zip(feature_idx, nonzero_mask, coef_abs) if keep]

    if not selected:
        return feature_idx  # fallback: keep all if Lasso zeroes everything

    # Cap before VIF — keep top features by |coefficient| to bound O(p²) cost.
    if len(selected) > _MAX_VIF_FEATURES:
        selected.sort(key=lambda t: t[1], reverse=True)
        selected = selected[:_MAX_VIF_FEATURES]

    return [idx for idx, _ in selected]


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
    """Iteratively remove the feature with highest VIF until all are below threshold.

    Capped at _MAX_VIF_ITERATIONS removals to prevent hangs on wide feature sets.
    """
    if len(feature_idx) <= 1:
        return feature_idx

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 3:
        return feature_idx

    Xv = X[mask].copy()
    idx_list = list(range(len(feature_idx)))

    for _ in range(_MAX_VIF_ITERATIONS):
        if len(idx_list) <= 1:
            break
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

    # Step 3: L1 / Lasso (caps at _MAX_VIF_FEATURES before VIF)
    X_sub = X_train[:, idx]
    idx = _select_by_lasso(X_sub, y_train, idx, random_state)
    if not idx:
        return [], []

    # Step 4: VIF (capped at _MAX_VIF_ITERATIONS removals)
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

    # Refit OLS (LinearRegression) on features selected by Lasso
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
    # Determine shapes from the first sample
    sample_shape = np.asarray(train_samples[0][0], dtype=float).shape
    n_rows = sample_shape[0] if len(sample_shape) > 1 else 1
    n_cols = sample_shape[1] if len(sample_shape) > 1 else sample_shape[0]

    y_train = np.array([np.asarray(s[1], dtype=float).flatten() for s in train_samples])
    y_test = np.array([np.asarray(s[1], dtype=float).flatten() for s in test_samples])
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    n_outputs = y_train.shape[1]
    predictions = np.full_like(y_test, np.nan)
    all_meta = []

    # Base feature names (one per input column, before time-step expansion)
    base_names = feature_names if feature_names is not None else [f"f{i}" for i in range(n_cols)]

    for j in range(n_outputs):
        # --- Step 0: Spearman pre-filter on base columns ---
        keep_cols, spearman_results = _prefilter_by_spearman(
            train_samples, j, base_names,
        )

        if verbose:
            print(f"[MLR] Spearman pre-filter: {len(keep_cols)}/{len(base_names)} "
                  f"base columns pass (p<0.05, |ρ|>0.25)")

        # Flatten only the surviving base columns
        def _flatten_filtered(samples):
            """Flatten each sample keeping only *keep_cols* columns."""
            rows = []
            for s in samples:
                X = np.asarray(s[0], dtype=float)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                rows.append(X[:, keep_cols].flatten())
            return np.array(rows, dtype=float)

        X_train_f = _flatten_filtered(train_samples)
        X_test_f = _flatten_filtered(test_samples)

        # Build flattened feature names for the filtered columns
        filtered_names = [
            f"{base_names[c]}_r{r}" for r in range(n_rows) for c in keep_cols
        ]

        pred_j, meta_j = fit_and_predict(
            X_train_f, y_train[:, j], X_test_f,
            feature_names=filtered_names, verbose=verbose,
        )
        predictions[:, j] = pred_j

        # Record Spearman filter results in metadata
        meta_j["spearman_kept_columns"] = [base_names[c] for c in keep_cols]
        meta_j["spearman_results"] = spearman_results
        all_meta.append(meta_j)

    return predictions, y_test, all_meta
