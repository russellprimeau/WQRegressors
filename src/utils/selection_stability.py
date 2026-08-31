"""How much of a feature selection the search actually resolved.

A beam search returns one subset, and reporting it as *the* selected feature set
implies the search distinguished it from the alternatives. Frequently it did not:
subsets sharing fewer than half their features score within 0.01 of each other on
12-47 independent samples, so which one comes first is close to arbitrary. What is
stable, and what is worth reporting, is which features keep appearing among the
subsets that score at the top.

This module computes that from a search trace, so it can be applied to a finished
sweep as well as to a running one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def selection_stability_from_trace(
    trace_df: pd.DataFrame,
    tolerance: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """Per-feature retention frequency across the subsets the search could not separate.

    Args:
        trace_df: Search trace, needing at least `features` (pipe-joined names) and
            `objective` columns.
        tolerance: Width of the near-optimal band, in objective units. Every candidate
            whose objective is within this of the best one is counted.

    Returns:
        `(frequency_table, summary)`. The table has one row per feature, ordered by
        retention frequency descending; `summary` carries the band and its population.

    Example:
        A feature at `retention_frequency` 1.0 appears in every near-optimal subset and
        is a finding; one at 0.4 is a coin flip the search did not resolve, and saying
        it was "selected" claims more than the evidence supports.
    """
    if trace_df is None or trace_df.empty:
        return pd.DataFrame(), {"n_near_optimal": 0, "n_evaluated": 0}
    if "features" not in trace_df.columns or "objective" not in trace_df.columns:
        raise ValueError("Trace needs both 'features' and 'objective' columns.")

    df = trace_df.copy()
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df = df[np.isfinite(df["objective"])]
    if df.empty:
        return pd.DataFrame(), {"n_near_optimal": 0, "n_evaluated": 0}

    best_obj = float(df["objective"].min())
    near = df[df["objective"] <= best_obj + float(tolerance)]
    subsets = [
        [f for f in str(row).split("|") if f.strip()]
        for row in near["features"].tolist()
    ]
    n_near = len(subsets)

    best_row = df.loc[df["objective"].idxmin()]
    best_features = {f for f in str(best_row["features"]).split("|") if f.strip()}

    # Every feature the near-optimal subsets mention, plus any the best subset uses,
    # so a feature that the band never retains still appears with a frequency of 0
    # rather than vanishing from the report.
    universe: list[str] = []
    for feats in subsets + [sorted(best_features)]:
        for f in feats:
            if f not in universe:
                universe.append(f)

    rows = []
    for feat in universe:
        hits = sum(1 for feats in subsets if feat in feats)
        rows.append({
            "feature": feat,
            "times_retained": int(hits),
            "n_near_optimal": int(n_near),
            "retention_frequency": (float(hits) / n_near) if n_near else float("nan"),
            "in_best_subset": bool(feat in best_features),
        })

    table = pd.DataFrame(rows).sort_values(
        ["retention_frequency", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    summary = {
        "n_evaluated": int(len(df)),
        "n_near_optimal": int(n_near),
        "best_objective": best_obj,
        "objective_band": float(tolerance),
        "n_features_best_subset": int(len(best_features)),
        "n_features_always_retained": int((table["retention_frequency"] >= 1.0).sum()),
        "n_features_never_retained": int((table["retention_frequency"] <= 0.0).sum()),
    }
    return table, summary
