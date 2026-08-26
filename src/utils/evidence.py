"""Statistical evidence for forecast-skill claims at small sample sizes.

This module replaces an earlier 0--5 "evidence score" that summed five gates:
bootstrap lower confidence bound, Diebold--Mariano, Wilcoxon signed-rank, sign
test, and interval coverage. That construction did not survive scrutiny on this
dataset, where a target has between 5 and 48 independent laboratory
measurements in the holdout period:

* Four of the five gates tested the same grouped loss differential with
  successively weaker assumptions, so the score summed one piece of evidence
  four times and implied a resolution the data cannot support.
* The coverage gate built its interval as ``+/- quantile(|residual|, 1-alpha)``
  from the residuals it then scored, making coverage an in-sample identity that
  could not fail. It contributed a free point to almost every target.
* At ``n = 5`` the sign test and Wilcoxon signed-rank cannot reach
  ``alpha = 0.05`` at all: the smallest attainable two-sided p is
  ``2 * 2**-5 = 0.0625``. Two of the five gates were unreachable a priori.
* Diebold--Mariano used the asymptotic normal with no small-sample correction,
  so it was the only gate that fired at ``n = 5`` and also the least reliable.

What replaces it is deliberately smaller: one effect size, one interval on that
effect size, one assumption-free consistency test, and an explicit statement of
whether the sample size permits any conclusion at all. Those four quantities
collapse into a four-level ordinal verdict.

The guiding rule is that a claim is only reported as supported when the sample
size could have refuted it.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - scipy is expected but not required for every path
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


# Verdict levels, weakest first. Ordering matters: aggregating across several
# reference forecasts takes the weakest verdict, so a claim must hold against
# every reference to survive.
NOT_SUPPORTED = "not_supported"
UNDERPOWERED = "underpowered"
DIRECTIONAL = "directional"
SUPPORTED = "supported"

VERDICT_ORDER = (NOT_SUPPORTED, UNDERPOWERED, DIRECTIONAL, SUPPORTED)
_VERDICT_RANK = {name: i for i, name in enumerate(VERDICT_ORDER)}

VERDICT_LABELS = {
    NOT_SUPPORTED: "Not supported",
    UNDERPOWERED: "Underpowered",
    DIRECTIONAL: "Directional",
    SUPPORTED: "Supported",
}

VERDICT_DEFINITIONS = {
    SUPPORTED: (
        "Positive skill, a bootstrap interval excluding zero, and a paired sign "
        "test significant at alpha, on a sample large enough to have failed."
    ),
    DIRECTIONAL: (
        "Positive skill and a majority of paired wins, but the sample does not "
        "reject the null at alpha."
    ),
    UNDERPOWERED: (
        "Positive skill and a majority of paired wins, but no outcome at this "
        "sample size could reach alpha, so significance was never attainable."
    ),
    NOT_SUPPORTED: "Skill is not positive, or the model loses the majority of pairs.",
}


def min_attainable_two_sided_p(n: int) -> float:
    """Smallest two-sided exact binomial p-value reachable with *n* paired trials.

    A unanimous result gives ``2 * 0.5**n``. Below ``n = 6`` this exceeds 0.05,
    so a sign test on five pairs cannot reach conventional significance no
    matter how one-sided the outcome is. Returns NaN for ``n < 1``.
    """
    n = int(n)
    if n < 1:
        return float("nan")
    return min(1.0, 2.0 * (0.5 ** n))


def power_is_attainable(n: int, alpha: float = 0.05) -> bool:
    """True when *n* paired trials could in principle reach *alpha*."""
    p_min = min_attainable_two_sided_p(n)
    return bool(np.isfinite(p_min) and p_min < float(alpha))


def exact_sign_test(diff: np.ndarray) -> dict:
    """Exact paired sign test on a loss differential.

    *diff* is model loss minus reference loss per group, so a negative entry is
    a win for the model. Ties are discarded, which is the standard treatment and
    keeps the null exact. The p-value is the two-sided exact binomial
    probability -- not a normal approximation -- so it is valid at any n.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    wins = int(np.sum(d < 0))
    losses = int(np.sum(d > 0))
    n = wins + losses
    out = {
        "wins": wins,
        "losses": losses,
        "n_pairs": n,
        "win_rate": float(wins / n) if n else float("nan"),
        "p_value": float("nan"),
        "min_attainable_p": min_attainable_two_sided_p(n) if n else float("nan"),
        # The test is two-sided, so a small p says only that the win rate is not
        # 0.5. Which side it falls on is a different statement and a reportable
        # one: "loses significantly more pairs than it wins" is a stronger
        # finding than "not supported", and the verdict alone cannot express it.
        "direction": ("favours_model" if wins > losses
                      else "favours_reference" if losses > wins
                      else "tied"),
    }
    if n < 1:
        return out
    if _scipy_stats is not None and hasattr(_scipy_stats, "binomtest"):
        try:
            out["p_value"] = float(
                _scipy_stats.binomtest(wins, n=n, p=0.5, alternative="two-sided").pvalue
            )
            return out
        except Exception:
            pass
    # Exact two-sided binomial without scipy: sum the tail probabilities of all
    # outcomes no more likely than the observed one.
    probs = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    observed = probs[wins]
    out["p_value"] = float(min(1.0, sum(p for p in probs if p <= observed + 1e-15)))
    return out


def dm_test(diff: np.ndarray, max_lag: int = 1, min_n: int = 30) -> dict:
    """Diebold--Mariano with the Harvey--Leybourne--Newbold small-sample fix.

    The uncorrected statistic referenced against the normal distribution
    over-rejects badly at the sample sizes available here; on five paired
    observations it returned p = 0.0005 where an exact sign test on the same
    data returned 0.0625. This applies the HLN (1997) scale correction and
    references Student's t on ``n - 1`` degrees of freedom.

    Returns NaN below *min_n*, because no correction rescues an asymptotic test
    at single-digit sample sizes. Reported as a diagnostic only; it does not
    enter the verdict.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    out = {"stat": float("nan"), "p_value": float("nan"), "n": n, "corrected": False}
    if n < max(5, int(min_n)):
        return out

    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    h = int(max(0, min(int(max_lag), n - 1)))
    var_hac = gamma0
    for lag in range(1, h + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var_hac += 2.0 * (1.0 - lag / (h + 1.0)) * cov
    if not np.isfinite(var_hac) or var_hac <= 0:
        return out

    stat = float(mean_d / math.sqrt(var_hac / n))
    # Harvey-Leybourne-Newbold correction for a horizon of h + 1 steps.
    steps = h + 1
    adj_num = n + 1.0 - 2.0 * steps + (steps * (steps - 1.0)) / n
    if adj_num <= 0:
        return out
    stat *= math.sqrt(adj_num / n)
    if _scipy_stats is not None:
        p = float(2.0 * _scipy_stats.t.sf(abs(stat), df=n - 1))
    else:  # pragma: no cover
        p = float("nan")
    out.update(stat=stat, p_value=p, corrected=True)
    return out


def grouped_skill_bootstrap(
    losses_model: np.ndarray,
    losses_ref: np.ndarray,
    group_ids,
    n_boot: int = 2000,
    seed: int = 42,
    block_len: int = 3,
    moving_block: bool = True,
) -> dict:
    """Percentile bootstrap for RMSE skill, resampling whole groups.

    *losses_model* and *losses_ref* are per-observation squared errors. Skill is
    recomputed from scratch on each resample, so the point estimate returned
    here and its interval always describe the same quantity -- the previous
    implementation reported a point estimate and an interval computed on
    different alignments, which let the lower bound exceed the point estimate.

    ``block_len`` is clamped below the number of groups. Left unclamped, a block
    length at or above the group count makes every moving-block resample a
    rotation of the entire sample, so all replicates are identical and the
    interval collapses to zero width -- which is what produced spurious
    "significant" lower bounds at n = 5.
    """
    lm = np.asarray(losses_model, dtype=float).reshape(-1)
    lr = np.asarray(losses_ref, dtype=float).reshape(-1)
    gids = list(group_ids)
    n = min(lm.size, lr.size, len(gids))
    out = {
        "skill": float("nan"),
        "ci05": float("nan"),
        "ci95": float("nan"),
        "prob_skill_gt0": float("nan"),
        "n_groups": 0,
        "n_boot_ok": 0,
        "block_len_used": int(max(1, block_len)),
        "block_len_clamped": False,
        "degenerate": False,
    }
    if n < 1:
        return out
    lm, lr, gids = lm[:n], lr[:n], gids[:n]
    finite = np.isfinite(lm) & np.isfinite(lr)
    if not np.any(finite):
        return out

    order: list = []
    group_to_idx: dict = {}
    for i in range(n):
        if not finite[i]:
            continue
        g = gids[i]
        if g not in group_to_idx:
            group_to_idx[g] = []
            order.append(g)
        group_to_idx[g].append(i)
    n_groups = len(order)
    out["n_groups"] = n_groups
    if n_groups < 1:
        return out

    def _skill(idx) -> float:
        rmse_m = math.sqrt(float(np.mean(lm[idx])))
        rmse_r = math.sqrt(float(np.mean(lr[idx])))
        if not np.isfinite(rmse_m) or not np.isfinite(rmse_r) or rmse_r <= 0:
            return float("nan")
        return 1.0 - rmse_m / rmse_r

    all_idx = [i for g in order for i in group_to_idx[g]]
    out["skill"] = _skill(all_idx)

    if n_groups < 3:
        # Too few groups to resample meaningfully; report the point estimate
        # with no interval rather than a fabricated one.
        return out

    # The block must also stay small relative to the group count. `assess`
    # already aggregates the loss differential to one value per independent
    # laboratory sample, so there is little serial dependence left for a block
    # to preserve, while a block spanning a large fraction of the groups makes
    # every resample a near-rotation of the whole sample and narrows the
    # interval artificially: at n = 5 a block of 3 gave roughly half the width
    # of i.i.d. group resampling, and the two agree by n = 48. Allowing at most
    # one block per four groups collapses to i.i.d. resampling exactly where the
    # sample is too small to estimate dependence, and is inert above n = 12.
    eff_block = int(max(1, min(int(block_len), n_groups // 4, n_groups - 1)))
    out["block_len_used"] = eff_block
    out["block_len_clamped"] = eff_block != int(max(1, block_len))

    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(max(1, n_boot))):
        if moving_block and eff_block > 1:
            chosen = []
            while len(chosen) < n_groups:
                start = int(rng.integers(0, n_groups))
                for j in range(eff_block):
                    chosen.append(order[(start + j) % n_groups])
                    if len(chosen) >= n_groups:
                        break
        else:
            chosen = [order[k] for k in rng.integers(0, n_groups, size=n_groups)]
        idx = [i for g in chosen for i in group_to_idx[g]]
        if not idx:
            continue
        s = _skill(idx)
        if np.isfinite(s):
            vals.append(s)

    arr = np.asarray(vals, dtype=float)
    out["n_boot_ok"] = int(arr.size)
    if arr.size:
        lo = float(np.quantile(arr, 0.05))
        hi = float(np.quantile(arr, 0.95))
        # Clamping the block length stops the interval collapsing in the common
        # case, but it can still collapse -- few groups, or a resample space too
        # small to vary the skill. A zero-width interval is not a 90% interval,
        # and reporting its lower edge as a confidence bound is how a lower
        # bound of 0.61 came to be published beside a point estimate of 0.05.
        # Emit nothing rather than a bound the resampling did not establish.
        spread = hi - lo
        scale = max(abs(lo), abs(hi), 1e-12)
        if not np.isfinite(spread) or spread <= 1e-9 * scale:
            out["degenerate"] = True
            out["prob_skill_gt0"] = float("nan")
        else:
            out["ci05"] = lo
            out["ci95"] = hi
            out["prob_skill_gt0"] = float(np.mean(arr > 0.0))
    return out


def classify(
    skill: float,
    ci05: float,
    win_rate: float,
    sign_p: float,
    n_pairs: int,
    alpha: float = 0.05,
) -> str:
    """Map the four reported quantities onto a verdict level."""
    if not np.isfinite(skill) or skill <= 0:
        return NOT_SUPPORTED
    if not np.isfinite(win_rate) or win_rate <= 0.5:
        return NOT_SUPPORTED
    if not power_is_attainable(n_pairs, alpha):
        return UNDERPOWERED
    significant = (
        np.isfinite(sign_p)
        and sign_p < float(alpha)
        and np.isfinite(ci05)
        and ci05 > 0.0
    )
    return SUPPORTED if significant else DIRECTIONAL


def weakest(verdicts) -> str:
    """Weakest verdict in *verdicts*; a claim must hold against every reference."""
    ranked = [v for v in verdicts if v in _VERDICT_RANK]
    if not ranked:
        return NOT_SUPPORTED
    return min(ranked, key=lambda v: _VERDICT_RANK[v])


def assess(
    losses_model: np.ndarray,
    losses_ref: np.ndarray,
    group_ids,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
    block_len: int = 3,
    moving_block: bool = True,
    dm_max_lag: int = 1,
) -> dict:
    """Full assessment of one model against one reference forecast.

    *losses_model* and *losses_ref* are per-observation squared errors on a
    common aligned evaluation set; *group_ids* identifies the independent
    sampling unit. Everything reported is derived from this single alignment.
    """
    boot = grouped_skill_bootstrap(
        losses_model, losses_ref, group_ids,
        n_boot=n_boot, seed=seed, block_len=block_len, moving_block=moving_block,
    )

    # Aggregate the loss differential to one value per group before testing, so
    # the paired tests see independent units rather than correlated rows.
    lm = np.asarray(losses_model, dtype=float).reshape(-1)
    lr = np.asarray(losses_ref, dtype=float).reshape(-1)
    gids = list(group_ids)
    n = min(lm.size, lr.size, len(gids))
    per_group: dict = {}
    for i in range(n):
        if not (np.isfinite(lm[i]) and np.isfinite(lr[i])):
            continue
        per_group.setdefault(gids[i], []).append(lm[i] - lr[i])
    diff = np.asarray([float(np.mean(v)) for v in per_group.values()], dtype=float)

    sign = exact_sign_test(diff)
    dm = dm_test(diff, max_lag=dm_max_lag)
    verdict = classify(
        boot["skill"], boot["ci05"], sign["win_rate"], sign["p_value"],
        sign["n_pairs"], alpha=alpha,
    )
    return {
        "skill": boot["skill"],
        "skill_ci05": boot["ci05"],
        "skill_ci95": boot["ci95"],
        "prob_skill_gt0": boot["prob_skill_gt0"],
        "n_groups": boot["n_groups"],
        "n_boot_ok": boot["n_boot_ok"],
        "block_len_used": boot["block_len_used"],
        "block_len_clamped": boot["block_len_clamped"],
        "bootstrap_degenerate": boot["degenerate"],
        "sign_direction": sign["direction"],
        "sign_wins": sign["wins"],
        "sign_losses": sign["losses"],
        "sign_n_pairs": sign["n_pairs"],
        "sign_win_rate": sign["win_rate"],
        "sign_p": sign["p_value"],
        "min_attainable_p": sign["min_attainable_p"],
        "power_attainable": power_is_attainable(sign["n_pairs"], alpha),
        "dm_stat": dm["stat"],
        "dm_p": dm["p_value"],
        "dm_applicable": dm["corrected"],
        "verdict": verdict,
        "alpha": float(alpha),
    }
