"""Emit the per-target results table for the manuscript.

Reads ``common_set_metrics.csv`` (written by ``z8_CommonSetMetrics.py``), in
which every method for a target is scored on the same test segments. That is the
only basis on which the two questions below can be answered side by side, because
an R^2 computed on 5 segments and an R^2 computed on 22 are answers to different
questions.

1. *Which method predicted the target best?*  Whichever of the seven method
   families achieved the highest R^2. The study asks whether the predictors carry
   information about the target, not only whether a machine-learning model can
   encode it, so a win by a statistical model is a result rather than a failure.
   Those wins are marked.

2. *Does that result stand out, or is it merely the best of several comparable
   ones?*  The skill score and the verdict from ``utils.evidence``, reported for
   **every** target. The bootstrap interval is what carries the answer: above
   zero the winner stands out, spanning zero it cannot be separated from the
   alternatives at that sample size, below zero the statistical model is
   reliably better. "Underpowered" is distinct from "not supported" -- it means
   no outcome at that sample size could have reached significance.

Targets are ordered by the best R^2 achieved, with no banding: a discrete
threshold would be invented rather than measured.

The holdout exposure behind each reported R^2 is printed to stdout rather than
carried as a table column, because it is the same story for every target and
belongs in the body text: the feature search's objective is test-split R^2, so
the candidate pool is itself the product of a few hundred consultations of the
test segments, and the retained configurations of the winning family are then
scored on those same segments and the best reported.

Usage:
    python src/z6_TargetSummaryTable.py
    python src/z6_TargetSummaryTable.py --summary <path> --output <path.tex>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.names import clean_target_label
from utils import evidence as ev
from utils import run_paths as rp

SUMMARY_NAME = "common_set_metrics.csv"
TABLE_DIR = Path("docs/report/draft/tables")

VERDICT_UNAVAILABLE = "__pending__"

VERDICT_TEX = {
    ev.SUPPORTED: "Supported",
    ev.DIRECTIONAL: "Directional",
    ev.UNDERPOWERED: "Underpowered",
    ev.NOT_SUPPORTED: "Not supported",
    VERDICT_UNAVAILABLE: r"\emph{pending}",
}

ML_FAMILIES = {"GP", "XGB", "Transformer"}

# The manuscript body is written in pure ASCII with LaTeX escapes (\AA, \upmu,
# \o), so generated tables must match that convention: a literal U+00B0 or a
# non-breaking space reaches inputenc as a raw byte and is a common source of
# build failures on the MDPI class.
# Written as explicit escapes: several of these are invisible or
# indistinguishable in a source listing, and a literal non-breaking space keyed
# by eye is how one slipped through and reached inputenc as a raw byte.
_TEX_CHAR_MAP = {
    "°": r"$^\circ$",    # degree sign
    " ": "~",            # non-breaking space
    "–": "--",           # en dash
    "—": "---",          # em dash
    "‘": "`",            # left single quote
    "’": "'",            # right single quote
    "“": "``",           # left double quote
    "”": "''",           # right double quote
    "µ": r"$\upmu$",     # micro sign
    "μ": r"$\upmu$",     # greek small letter mu
}


def _tex_safe(text: str) -> str:
    out = "".join(_TEX_CHAR_MAP.get(ch, ch) for ch in str(text))
    residual = sorted({ch for ch in out if ord(ch) > 127})
    if residual:
        raise ValueError(
            "Unmapped non-ASCII character(s) in label %r: %s. Add them to "
            "_TEX_CHAR_MAP rather than emitting raw bytes into LaTeX."
            % (text, [hex(ord(c)) for c in residual])
        )
    return out


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")


def _fmt(val, places=3):
    if val is None or pd.isna(val):
        return "---"
    return f"{val:.{places}f}"


def _fmt_int(val):
    if val is None or pd.isna(val):
        return "---"
    return f"{int(round(val))}"


def _fmt_skill(ss, lo, hi):
    """Effect size with its interval, or the estimate alone when the bootstrap
    could not establish one (too few groups, or a degenerate resample)."""
    if pd.isna(ss):
        return "---"
    if pd.isna(lo) or pd.isna(hi):
        return f"{ss:.2f}"
    return f"{ss:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_p(p, p_min):
    """Sign-test p with the smallest value attainable at this n beside it.

    A p-value cannot be read without its floor: where p_min exceeds alpha no
    outcome could have been significant, and reporting the p alone would invite
    that case to be read as a tested negative.
    """
    if pd.isna(p):
        return "---"
    if pd.isna(p_min):
        return f"{p:.2f}"
    return f"{p:.2f} ({p_min:.2f})"


def build_rows(df: pd.DataFrame, prefix: str) -> list[dict]:
    required = {"best_family", "best_r2", "n_common"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"summary CSV is missing {sorted(missing)}; this script expects the output of "
            "z8_CommonSetMetrics.py, in which every method is scored on the same segments."
        )
    has_verdicts = "aligned_verdict" in df.columns
    if not has_verdicts:
        print("[WARN] No 'aligned_verdict' column: the Verdict column is emitted as "
              "'\\emph{pending}' so it cannot be mistaken for a finding.")

    rows = []
    for _, r in df.iterrows():
        family = str(r.get("best_family", "") or "").strip()
        is_ref = family not in ML_FAMILIES
        key = family.lower().replace("-", "")

        verdict = (str(r.get("aligned_verdict") or ev.NOT_SUPPORTED)
                   if has_verdicts else VERDICT_UNAVAILABLE)

        rows.append(dict(
            target=_tex_safe(clean_target_label(str(r.get("dataset", "")), prefix)),
            method=_tex_safe(family or "---"),
            is_reference=is_ref,
            r2=_f(r, "best_r2"),
            n_candidates=_f(r, f"{key}_n_candidates"),
            r2_median=_f(r, f"{key}_r2_median"),
            n_search=_f(r, "n_search_holdout_scorings"),
            n=_f(r, "n_common"),
            # Skill measures whether a result stands out from the alternatives or is
            # merely the best of several comparable ones, which is a question for
            # every target regardless of which family won it. Where a reference
            # forecast won, the skill of the best learned model against it is
            # negative, and the interval says whether that deficit is resolvable.
            skill=_f(r, "skill_vs_best_ref"),
            lo=_f(r, "aligned_skill_ci05"),
            hi=_f(r, "aligned_skill_ci95"),
            p=_f(r, "aligned_sign_p"),
            p_min=_f(r, "aligned_min_attainable_p"),
            verdict=verdict,
        ))

    rows.sort(key=lambda d: -(d["r2"] if pd.notna(d["r2"]) else -9e9))
    return rows


def render(rows: list[dict]) -> str:
    out = [
        r"\begin{table}[H]",
        r"\caption{Per-target accuracy and statistical support on the common "
        r"evaluation set.\label{tab:targets}}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{"
        # tabularx: the \hsize coefficients must sum to the number of X columns
        # (7 here), or the table is set to the wrong total width and the columns
        # run over one another.  1.75 + 1.0 + 0.5 + 0.35 + 1.55 + 0.85 + 1.0 = 7.0
        r">{\hsize=1.75\hsize\raggedright\arraybackslash}X"
        r">{\hsize=1.0\hsize\raggedright\arraybackslash}X"
        r">{\hsize=0.5\hsize\centering\arraybackslash}X"
        r">{\hsize=0.35\hsize\centering\arraybackslash}X"
        r">{\hsize=1.55\hsize\centering\arraybackslash}X"
        r">{\hsize=0.85\hsize\centering\arraybackslash}X"
        r">{\hsize=1.0\hsize\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Target} & \textbf{Best method} & \textbf{$R^2$} & "
        r"\textbf{$n$} & \textbf{SS (95\% CI)} & "
        r"\textbf{$p$ ($p_{\min}$)} & \textbf{Verdict}\\",
        r"\midrule",
    ]
    for row in rows:
        marker = r"$^\dagger$" if row["is_reference"] else ""
        verdict = "---" if row["verdict"] is None else VERDICT_TEX.get(row["verdict"], "---")
        out.append(
            "%s & %s%s & %s & %s & %s & %s & %s\\\\"
            % (
                row["target"],
                row["method"],
                marker,
                _fmt(row["r2"]),
                _fmt_int(row["n"]),
                _fmt_skill(row["skill"], row["lo"], row["hi"]),
                _fmt_p(row["p"], row["p_min"]),
                verdict,
            )
        )
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=None,
                    help="Common-set metrics CSV. Defaults to the reporting root's.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to tables/target_summary.tex for the reporting root and "
                         "tables/target_summary_<root>.tex for any other arm, so a second "
                         "arm cannot silently replace the manuscript's main table.")
    ap.add_argument("--dataset-prefix", type=str, default="MC")
    args = ap.parse_args()

    summary = rp.resolve_output(args.summary, None, SUMMARY_NAME)
    if not summary.exists():
        raise SystemExit(f"summary CSV not found: {summary}")

    # The table filename carries the identity of the arm it came from. Both arms
    # are reported -- the reporting root in the body, the other in supplementary --
    # and they must not resolve to the same file.
    root = rp.root_of_summary(summary)
    if args.output is not None:
        output = args.output
    elif rp.is_reporting_root(root):
        output = rp.REPO_ROOT / TABLE_DIR / "target_summary.tex"
    else:
        output = rp.REPO_ROOT / TABLE_DIR / f"target_summary_{root.name}.tex"

    df = pd.read_csv(summary)
    rows = build_rows(df, args.dataset_prefix)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows), encoding="utf-8")
    print(f"[INFO] Wrote {output} ({len(rows)} targets) from {root.name}")

    # Counts only: targets have unrelated dynamics, so nothing is averaged across
    # them.
    n_ref = sum(1 for r in rows if r["is_reference"])
    print(f"  a statistical model was the best predictor for {n_ref} of {len(rows)} targets")

    # The selection exposure is the same story for every target, so it is reported
    # here for the body text rather than as a table column.
    cand = [r["n_candidates"] for r in rows if pd.notna(r["n_candidates"])]
    med = [r["r2_median"] for r in rows if pd.notna(r["r2_median"])]
    srch = {int(r["n_search"]) for r in rows if pd.notna(r["n_search"])}
    if cand:
        print(f"  selection exposure: winner was best of {int(min(cand))}-{int(max(cand))} "
              f"configurations; their median R2 spans {min(med):+.2f} to {max(med):+.2f}")
    if srch:
        print(f"  search scorings per target: {sorted(srch)}")
    for v in list(reversed(ev.VERDICT_ORDER)) + [VERDICT_UNAVAILABLE]:
        n = sum(1 for r in rows if r["verdict"] == v)
        if n:
            print(f"  {n:2d}  verdict: {v}")
    n_blank = sum(1 for r in rows if r["verdict"] is None)
    if n_blank:
        print(f"  {n_blank:2d}  no verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
