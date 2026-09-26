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

`mc_replicates/` holds one file per window per replicate for the families that receive input
uncertainty as data — **MLR, XGBoost and the transformer**; the GP reads `samples` because it
marginalises that uncertainty inside its kernel. Two rules govern what is written. A root
whose predictors include nothing carrying a measured uncertainty distribution gets no
`mc_replicates/` at all, and its configs read `samples` instead. Within a root that does, a
window whose uncertainty-bearing channels are all NaN gets **one** file rather than K, since
its replicates would be identical copies — one rather than none, because this folder is the
training set and a window with no file here would drop out of training. The offsets actually
applied are recorded in `<target>/provenance/mc_offsets.csv`, which makes the tree
reconstructible from `samples/` rather than being its own only record.

`gp_model.pt` is written at schema version 3. Versions 1-2 stored the per-feature input
uncertainty twice -- once at top level and once as a persistent kernel buffer inside
`model_state_dict` -- each copy tiled across the input window, which made 89% of every
artifact one repeated array. Version 3 stores the arrays in their pre-tile form with a
repeat count and does not persist the buffers, which is ~97% smaller for the same fit.
`utils.gp_utils.uncertainty_arrays` reads either schema, so older roots load unchanged;
`v5_CheckArtifactRoundTrip.py` is the gate that keeps it that way.

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
    --data-root data/output/CV19 --all-datasets
#    MLR is a machine-learning family, not a reference forecast: it reads the
#    predictors, where naive, seasonal and linear read only the target history.
#    Passing --treat-mlr-as-baseline moves it to the reference set and drops its
#    variants from the ML comparison figure.

# 5. Manifest compliance. Zero errors before anything is reported.
.venv/Scripts/python src/validate_run_outputs.py --root data/output/CV19

# 5b. Saved models reload. The GP artifact is the only one whose reconstruction can
#     fail silently -- o_PredictionTimeseries turns a load error into a [WARN] and a
#     missing figure -- so this is checked rather than assumed.
.venv/Scripts/python src/v5_CheckArtifactRoundTrip.py --root data/output/CV19

# 5c. Monte Carlo replicates. Run this on any root generated after 2026-09-23; roots
#     written before then report "legacy replicate tree" per target, which is a
#     statement about their age rather than a defect.
.venv/Scripts/python src/v5_CheckArtifactRoundTrip.py --replicates --root data/output/CV19

# 6. Canonical results: every method on one evaluation set per target.
.venv/Scripts/python src/z8_CommonSetMetrics.py --root data/output/CV19

# 6b. Seed robustness. REQUIRED, not optional: steps 1-6 fit one seed per model, and
#     the measured seed spread reaches 0.44 R^2 on XGBoost and 0.42 on the transformer.
#     Three of five CV22 XGBoost wins did not survive seed averaging, so this decides
#     which model is reported, not merely how precisely.
#
#     Run the revert FIRST on any root that has been post-processed before. It restores
#     each run's predictions_seed0.csv and is a no-op otherwise. Skipping it on a
#     re-run does not fail: z8 reads the already-ensembled vector as the single-seed
#     baseline, v3 then finds seed 0 does not reproduce it and marks the candidate
#     non-reproducing, and z17 skips every such candidate. The root ends up reported as
#     seed-ensembled while still carrying single-seed numbers.
.venv/Scripts/python src/z17_ApplySeedEnsembles.py --root data/output/CV19 --revert
.venv/Scripts/python src/v3_SeedVarianceRefit.py   --root data/output/CV19 --seeds 6
.venv/Scripts/python src/z17_ApplySeedEnsembles.py --root data/output/CV19 --seeds 6
.venv/Scripts/python src/z8_CommonSetMetrics.py    --root data/output/CV19
#     v3 refits only candidates within a band measured from each family's own seed
#     spread, and reuses any seed fit already on disk, so an interrupted refit resumes
#     rather than restarting. z17 installs each candidate's mean prediction vector; the
#     second z8 is what makes the reported score come from that vector.

# 7. Manuscript outputs. z6 writes into the manuscript repository, which is a
#    sibling checkout, not a submodule: set WQ_MANUSCRIPT_ROOT first (or run from
#    wq-forecasting.code-workspace, which sets it) or the script exits without
#    writing.
.venv/Scripts/python src/z6_TargetSummaryTable.py
.venv/Scripts/python src/z9_QualityMatrix.py
.venv/Scripts/python src/z7_StructureCompare.py --exclude-model none
```

Steps 2, 5 and 5b are gates: if any reports errors, the results are not reportable and the
cause must be fixed before step 7. Step 6b is required before step 7 for the same reason:
skipping it reports one draw of a stochastic fit as though it were the model's performance. `z8_CommonSetMetrics.py` is the canonical source for the
manuscript's numbers — `z1`'s own best-model selection is scored on configuration-specific
evaluation sets and is retained for the sweep-level figures only.
