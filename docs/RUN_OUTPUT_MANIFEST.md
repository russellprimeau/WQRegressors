# Sweep output manifest

The sweep is expensive enough to be run once, so this records what it must emit and
which part of the analysis consumes each artifact. Anything listed here that is
missing or unattributable after a run means another run.

Enforced by `src/validate_run_outputs.py`, which fails on every **required** item
below. Run it against the output root before treating a sweep as finished:

```
.venv/Scripts/python src/validate_run_outputs.py --root data/output/<ROOT>
```

## Per target (`<ROOT>/MC_<target>/`)

| Artifact | Required | Consumed by |
| --- | --- | --- |
| `samples/segment_*.csv` | yes | everything; the target values and predictor windows |
| `normalization.json` | yes | returning targets to physical units; the target-definition check |
| `forecasts/feature_sweeps/feature_sweep_final_metrics.csv` | yes | best-model selection, Figures 5/7/8, the summary table |
| `forecasts/feature_sweeps/feature_search_trace_*.csv` | yes, exactly one | predictor membership per subset; the profiler classification |
| `forecasts/feature_sweeps/feature_selected_subsets_*.csv` | preferred | documenting which subsets were retained |

Constraints on `feature_sweep_final_metrics.csv`:

- **Exactly one `std_target` value per target**, positive and finite. NRMSE is only
  comparable between methods if its denominator describes the target rather than the
  configuration being scored. Multiple values mean the nRMSE figure is comparing
  quantities normalized by different sigmas.
- **`nrmse` populated for every row that has an `rmse`.** A null leaves that model
  silently absent from the nRMSE figure rather than visibly missing.
- Exactly one `row_count`, matching the search trace's `_r<N>` suffix. More than one
  means results from different window lengths are mixed.

## Per run (`.../feature_sweeps/<run_dir>/`)

| Artifact | Required | Consumed by |
| --- | --- | --- |
| `predictions.csv` | yes | the common evaluation set; all skill and paired statistics |
| `train_files.txt`, `test_files.txt` | yes | split auditing; training-set size; leakage checks |
| `evaluation_summary.csv` | yes | per-run metrics and the instrumentation below |
| `model_config.json` | yes | recording how the inputs were constructed |
| `training_stop_summary.json` | preferred | whether the model actually trained |

`predictions.csv` must carry `kind` and `sample_file`. The common evaluation set is
built by intersecting test `sample_file`s across runs, and a segment enters it only
if it is labelled `kind == "test"` in every run compared — which is what makes the
construction leakage-free. Without those two columns a run cannot participate.

`evaluation_summary.csv` must carry:

- **`n_train_samples`, non-null**, and agreeing with `train_files.txt`. Without it the
  training-set cost of a predictor choice cannot be read off the outputs at all.
- **`train_/test_n_dropped`, `_drop_rate`, and `_drop_predictors`.** Any run reporting
  dropped samples must name the predictor columns responsible. This is the rule that
  matters most: a single partial-coverage predictor can remove three quarters of a
  target's evaluation samples, and without attribution that loss is invisible.
- **`n_predictions_clipped`**, recording predictions constrained to the target's
  normalized support. A correction applied silently is worse than one recorded.

## Global constraints

- **No prediction outside `[0, 1]`.** Predictions are of min-max normalized targets, so
  a value outside that interval is extrapolation, not a forecast. One such point is
  enough to dominate a squared-error metric and decide which model is reported as best.
- **Every dropped sample is attributable** — to a reason, and where predictors are
  responsible, to the columns.
- **One target definition per root.** `src/v1_CheckTargetDefinition.py` must pass for
  every target: the differential target must be the difference against the window-start
  value, not against the latest measurement inside the window.

## Full run sequence

`d_RunResample.py` writes in place and does **not** clear the output root, so a rerun over an
existing tree would leave the previous `forecasts/` subtree — thousands of run directories —
mixed in with the new one. Archive first; the rename is instant on the same filesystem.

```bash
# 0. Archive the previous tree. Nothing else guarantees the run is not mixed.
mv data/output/CV19 data/output/CV19_superseded

# 1. Regenerate samples and model configs (emits the four GP variants).
.venv/Scripts/python src/d_RunResample.py --config data/input/splitting/resample_diff.yml

# 2. Cheap gate before committing hours: the differential target must be the
#    difference against the window-start value, for all 14 targets.
.venv/Scripts/python src/v1_CheckTargetDefinition.py

# 3. The sweep. This is the long step.
.venv/Scripts/python src/h_RunMCFeatureSelectionSweep.py \
    --data-root data/output/CV19 --limit-datasets 14

# 4. Post-processing: one sigma per target, NRMSE, evidence statistics.
.venv/Scripts/python src/z1_FeaturePostProcess.py \
    --data-root data/output/CV19 --all-datasets --treat-mlr-as-baseline

# 5. Manifest compliance. Zero errors before anything is reported.
.venv/Scripts/python src/validate_run_outputs.py --root data/output/CV19

# 6. Canonical results: every method on one evaluation set per target.
.venv/Scripts/python src/z8_CommonSetMetrics.py --root data/output/CV19

# 7. Manuscript outputs.
.venv/Scripts/python src/z6_TargetSummaryTable.py
.venv/Scripts/python src/z9_QualityMatrix.py
.venv/Scripts/python src/z7_StructureCompare.py --exclude-model none
```

Steps 2 and 5 are gates: if either reports errors, the results are not reportable and the
cause must be fixed before step 7. `z8_CommonSetMetrics.py` is the canonical source for the
manuscript's numbers — `z1`'s own best-model selection is scored on configuration-specific
evaluation sets and is retained for the sweep-level figures only.
