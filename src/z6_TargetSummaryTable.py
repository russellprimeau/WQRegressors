"""
Emit the per-target results table for the manuscript from a z1 summary CSV.

Reads ``summary_best_model_performance.csv`` (written by ``z1_FeaturePostProcess.py``)
and writes an MDPI-compatible LaTeX ``table`` environment.

Two independent questions are reported side by side, because they have different
answers and conflating them is what made earlier drafts misleading:

1. *Which method predicted the target best?*  Whichever of the best
   machine-learning model or the best reference forecast achieved the higher
   $R^2$. The study asks whether the predictors carry information about the
   target, not only whether a machine-learning model can encode it, so a
   reference win is a result rather than a failure. Reference wins are marked.

2. *Is the machine-learning advantage defensible?*  The verdict from
   ``utils.evidence``: supported, directional, underpowered, or not supported.
   "Underpowered" is distinct from "not supported" -- it means no outcome at
   that sample size could have reached significance.

Rows are grouped by the first question and the verdict column answers the second.

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

DEFAULT_SUMMARY = Path("data/output/CV19/summaries/summary_best_model_performance.csv")
DEFAULT_OUTPUT = Path("docs/report/draft/tables/target_summary.tex")

GROUP_ML = 0
GROUP_REF = 1
GROUP_NONE = 2
GROUP_HEADINGS = {
    GROUP_ML: "Machine learning predicted the target best",
    GROUP_REF: "A reference forecast predicted the target best",
    GROUP_NONE: "No method explained variance in the holdout period",
}

VERDICT_UNAVAILABLE = "__pending__"

VERDICT_TEX = {
    ev.SUPPORTED: "Supported",
    ev.DIRECTIONAL: "Directional",
    ev.UNDERPOWERED: "Underpowered",
    ev.NOT_SUPPORTED: "None",
    VERDICT_UNAVAILABLE: r"\emph{pending}",
}


# The manuscript body is written in pure ASCII with LaTeX escapes (\AA, \upmu,
# \o), so generated tables must match that convention: a literal U+00B0 or a
# non-breaking space reaches inputenc as a raw byte and is a common source of
# build failures on the MDPI class.
_TEX_CHAR_MAP = {
    "°": r"$^\circ$",
    " ": "~",
    "–": "--",
    "—": "---",
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "µ": r"$\upmu$",
    "μ": r"$\upmu$",
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
    """Effect size with its interval, or a dash when no interval was computable."""
    if pd.isna(ss):
        return "---"
    if pd.isna(lo) or pd.isna(hi):
        return f"{ss:.3f}"
    return f"{ss:.3f} [{lo:.2f}, {hi:.2f}]"


def build_rows(df: pd.DataFrame, prefix: str) -> list[dict]:
    has_verdicts = "verdict_overall" in df.columns
    if not has_verdicts:
        print(
            "[WARN] No 'verdict_overall' column in the summary CSV: it predates the "
            "support-verdict machinery. The Verdict column is emitted as "
            "'\\emph{pending}' so it cannot be mistaken for a finding. Re-run "
            "z1_FeaturePostProcess.py, then re-run this script."
        )
    rows = []
    for _, r in df.iterrows():
        ml_label = str(r.get("best_model_label", "") or "").strip()
        ref_label = str(r.get("best_baseline_label", "") or "").strip()
        ml_r2 = _f(r, "best_model_r2")
        ref_r2 = _f(r, "best_baseline_r2")
        skill = _f(r, "skill_vs_best_baseline")
        lo = _f(r, "skill_ci05_vs_best_baseline")
        hi = _f(r, "skill_ci95_vs_best_baseline")
        n_pairs = _f(r, "n_pairs_vs_best_baseline")
        if pd.isna(n_pairs) or n_pairs <= 0:
            n_pairs = _f(r, "n_test_independent_source")
        # A summary written before the verdict machinery existed has no verdict
        # column at all. Emitting the NOT_SUPPORTED default there would render as
        # a real finding of "no support"; mark it unavailable instead.
        if has_verdicts:
            verdict = str(r.get("verdict_overall", ev.NOT_SUPPORTED) or ev.NOT_SUPPORTED)
        else:
            verdict = VERDICT_UNAVAILABLE

        ref_wins = bool(pd.notna(ref_r2) and (pd.isna(ml_r2) or ref_r2 > ml_r2))
        best_label = ref_label if ref_wins else ml_label
        best_r2 = ref_r2 if ref_wins else ml_r2

        if pd.notna(best_r2) and best_r2 <= 0:
            group = GROUP_NONE
        elif ref_wins:
            group = GROUP_REF
        else:
            group = GROUP_ML

        # R^2 and SS should agree on which side won: on a common evaluation set a
        # higher R^2 implies a lower RMSE, hence positive skill. A disagreement
        # means the two were computed on different alignments, so the row cannot
        # be grouped reliably. Surface it rather than silently mis-filing it.
        if pd.notna(ml_r2) and pd.notna(ref_r2) and pd.notna(skill):
            if (ml_r2 > ref_r2) != (skill > 0):
                print(
                    f"[WARN] {r.get('dataset', '?')}: R^2 ordering and skill sign "
                    f"disagree (model R2={ml_r2:.3f}, reference R2={ref_r2:.3f}, "
                    f"SS={skill:+.3f}). These are computed on different aligned "
                    f"sets; the grouping of this row is not trustworthy."
                )

        rows.append(
            dict(
                target=_tex_safe(clean_target_label(str(r.get("dataset", "")), prefix)),
                method=_tex_safe(best_label or "---"),
                is_reference=ref_wins,
                r2=best_r2,
                skill=skill,
                lo=lo,
                hi=hi,
                n=n_pairs,
                verdict=verdict,
                group=group,
            )
        )
    rows.sort(key=lambda d: (d["group"], -(d["r2"] if pd.notna(d["r2"]) else -9e9)))
    return rows


def render(rows: list[dict]) -> str:
    out = [
        r"\begin{table}[H]",
        r"\caption{Best-performing method for each target on the chronological "
        r"holdout period, and whether any machine-learning advantage is "
        r"statistically defensible. The method with the higher $R^2$ is reported "
        r"whether it is a machine-learning model or a reference forecast; rows "
        r"marked $^\dagger$ were won by a reference forecast. SS is the skill of "
        r"the best machine-learning model against the strongest reference "
        r"(Equation~\ref{eq:skill}) with its 95\% bootstrap interval, so a "
        r"negative value means a reference was more accurate. $n$ is the number "
        r"of independent laboratory measurements in the holdout period. Verdict "
        r"is defined in Section~\ref{ch:EvaluationMetrics}; \emph{underpowered} "
        r"means no outcome at that $n$ could have reached "
        r"$\alpha=0.05$.\label{tab:targets}}",
        r"\begin{tabularx}{\textwidth}{"
        r">{\raggedright\arraybackslash}X l c l c l}",
        r"\toprule",
        r"\textbf{Target} & \textbf{Best method} & \textbf{$R^2$} & "
        r"\textbf{SS (95\% CI)} & \textbf{$n$} & \textbf{Verdict}\\",
    ]
    current = None
    for row in rows:
        if row["group"] != current:
            current = row["group"]
            out.append(r"\midrule")
            out.append(r"\multicolumn{6}{l}{\textit{%s}}\\" % GROUP_HEADINGS[current])
        marker = r"$^\dagger$" if row["is_reference"] else ""
        out.append(
            "%s & %s%s & %s & %s & %s & %s\\\\"
            % (
                row["target"],
                row["method"],
                marker,
                _fmt(row["r2"]),
                _fmt_skill(row["skill"], row["lo"], row["hi"]),
                _fmt_int(row["n"]),
                VERDICT_TEX.get(row["verdict"], "---"),
            )
        )
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dataset-prefix", type=str, default="MC")
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"summary CSV not found: {args.summary}")
    df = pd.read_csv(args.summary)
    rows = build_rows(df, args.dataset_prefix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(f"[INFO] Wrote {args.output} ({len(rows)} targets)")

    for g, head in GROUP_HEADINGS.items():
        print(f"  {sum(1 for r in rows if r['group'] == g):2d}  {head}")
    print(f"  reference forecast won on "
          f"{sum(1 for r in rows if r['is_reference'])} of {len(rows)} targets")
    for v in list(reversed(ev.VERDICT_ORDER)) + [VERDICT_UNAVAILABLE]:
        n = sum(1 for r in rows if r["verdict"] == v)
        if n:
            print(f"  {n:2d}  verdict: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
