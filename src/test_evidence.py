"""Checks for utils.evidence, including the specific defects it was written to fix."""
from __future__ import annotations

import numpy as np

from utils import evidence as ev


def _report(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))
    return bool(ok)


def test_attainability():
    print("attainability floor at small n")
    ok = True
    # 2 * 0.5**5 = 0.0625 > 0.05, so five pairs can never reach alpha.
    ok &= _report("n=5 min p is 0.0625", abs(ev.min_attainable_two_sided_p(5) - 0.0625) < 1e-12)
    ok &= _report("n=5 not attainable", not ev.power_is_attainable(5, 0.05))
    ok &= _report("n=6 attainable", ev.power_is_attainable(6, 0.05))
    # A unanimous 5-0 win must still fail, and must be labelled underpowered
    # rather than "not supported" -- the distinction the old score could not make.
    diff = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    s = ev.exact_sign_test(diff)
    ok &= _report("5-0 sweep gives p=0.0625", abs(s["p_value"] - 0.0625) < 1e-12,
                  f"win_rate={s['win_rate']}")
    v = ev.classify(skill=0.4, ci05=0.1, win_rate=1.0, sign_p=s["p_value"], n_pairs=5)
    ok &= _report("5-0 sweep classified underpowered", v == ev.UNDERPOWERED, v)
    return ok


def test_bootstrap_not_degenerate():
    print("bootstrap does not collapse when block_len >= n_groups")
    rng = np.random.default_rng(0)
    n = 5
    groups = [f"g{i}" for i in range(n)]
    lm = np.abs(rng.normal(1.0, 0.3, n))
    lr = np.abs(rng.normal(1.6, 0.3, n))
    # block_len 7 with 5 groups previously made every resample a rotation of the
    # whole sample, so ci05 == ci95 == a point mass.
    out = ev.grouped_skill_bootstrap(lm, lr, groups, n_boot=500, seed=1, block_len=7)
    ok = True
    ok &= _report("block_len clamped", out["block_len_clamped"] and out["block_len_used"] < n,
                  f"used={out['block_len_used']}")
    width = out["ci95"] - out["ci05"]
    ok &= _report("interval has non-zero width", np.isfinite(width) and width > 1e-9,
                  f"ci=[{out['ci05']:.4f}, {out['ci95']:.4f}]")
    ok &= _report("point estimate inside interval",
                  out["ci05"] - 1e-9 <= out["skill"] <= out["ci95"] + 1e-9,
                  f"skill={out['skill']:.4f}")
    return ok


def test_point_estimate_consistent_with_interval():
    print("point estimate and interval describe the same quantity")
    rng = np.random.default_rng(7)
    n = 40
    groups = [f"g{i}" for i in range(n)]
    lm = np.abs(rng.normal(1.0, 0.4, n))
    lr = np.abs(rng.normal(1.5, 0.4, n))
    out = ev.grouped_skill_bootstrap(lm, lr, groups, n_boot=800, seed=3, block_len=3)
    direct = 1.0 - np.sqrt(lm.mean()) / np.sqrt(lr.mean())
    ok = _report("skill matches direct computation", abs(out["skill"] - direct) < 1e-12,
                 f"{out['skill']:.6f} vs {direct:.6f}")
    ok &= _report("ci05 <= skill <= ci95",
                  out["ci05"] <= out["skill"] <= out["ci95"],
                  f"[{out['ci05']:.4f}, {out['skill']:.4f}, {out['ci95']:.4f}]")
    return ok


def test_direction_matters():
    print("a worse model is never credited")
    rng = np.random.default_rng(11)
    n = 40
    groups = [f"g{i}" for i in range(n)]
    # Model loss clearly larger than the reference: significantly worse.
    lm = np.abs(rng.normal(2.0, 0.3, n))
    lr = np.abs(rng.normal(1.0, 0.3, n))
    res = ev.assess(lm, lr, groups, n_boot=400, seed=5, block_len=3)
    ok = _report("skill is negative", res["skill"] < 0, f"{res['skill']:.4f}")
    ok &= _report("win rate below half", res["sign_win_rate"] < 0.5, f"{res['sign_win_rate']:.3f}")
    ok &= _report("verdict is not_supported", res["verdict"] == ev.NOT_SUPPORTED, res["verdict"])
    return ok


def test_dm_small_sample():
    print("Diebold-Mariano withheld at small n")
    rng = np.random.default_rng(2)
    small = ev.dm_test(rng.normal(-1.0, 0.2, 5))
    ok = _report("n=5 returns NaN", not np.isfinite(small["p_value"]))
    big = ev.dm_test(rng.normal(-1.0, 0.2, 60))
    ok &= _report("n=60 is computed and corrected",
                  np.isfinite(big["p_value"]) and big["corrected"],
                  f"p={big['p_value']:.5f}")
    return ok


def test_supported_case_and_aggregation():
    print("a genuine win is reported as supported; weakest verdict wins")
    rng = np.random.default_rng(4)
    n = 45
    groups = [f"g{i}" for i in range(n)]
    lm = np.abs(rng.normal(1.0, 0.15, n))
    lr = np.abs(rng.normal(2.2, 0.15, n))
    res = ev.assess(lm, lr, groups, n_boot=800, seed=9, block_len=3)
    ok = _report("verdict is supported", res["verdict"] == ev.SUPPORTED,
                 f"{res['verdict']} ss={res['skill']:.3f} p={res['sign_p']:.2g} "
                 f"ci05={res['skill_ci05']:.3f}")
    ok &= _report("weakest across references",
                  ev.weakest([ev.SUPPORTED, ev.DIRECTIONAL, ev.NOT_SUPPORTED]) == ev.NOT_SUPPORTED)
    ok &= _report("weakest of empty is not_supported", ev.weakest([]) == ev.NOT_SUPPORTED)
    return ok


def main():
    tests = [
        test_attainability,
        test_bootstrap_not_degenerate,
        test_point_estimate_consistent_with_interval,
        test_direction_matters,
        test_dm_small_sample,
        test_supported_case_and_aggregation,
    ]
    results = []
    for t in tests:
        results.append(bool(t()))
        print()
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} test groups passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
