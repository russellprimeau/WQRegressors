# Results series registry

Every top-level directory under `data/output/` is a *results series*: one resampling
configuration, swept and evaluated. This file is the authoritative record of what each
series is, what produced it, what consumes it, and what supersedes it.

It lives in `docs/` rather than in `data/output/` because `data/output/` is gitignored.
A registry that is itself untracked is lost on any fresh checkout, which is how the
archive became unauditable in the first place. Each series also carries its own
`README.md` and `provenance/resample_config.yml` beside the data, but those are local.

**When a new series is created, add a row here in the same commit.** The cost of not
doing so is not hypothetical: it is why 14.8 GB sat undisposable, because nothing
recorded whether anyone still needed it.

## Retention rule

Anything that is a source for content currently in the manuscript — figures, tables, or
values cited in the text — is retained **in full**, for traceability. Whether a
regeneration script would need only a summary file is immaterial: the claim in the paper
has to be traceable to the data it came from.

A series is released only when a *different, completed* dataset has superseded it as that
source. A series awaiting its superseding run is retained in full and marked `hold` below.

## Active series

| Series | Produced by | Status | Role and consumers |
| --- | --- | --- | --- |
| `CV22_profilerless` | `resample_diff_profilerless.yml` | **manuscript source** | `run_paths.REPORTING_ROOT` and `DEFAULT_ROOT`. Table 1 (`z6` variant `main`), `z9` quality matrix, `z10` predictor retention, `z12` method comparison, `z15` test-set predictions, `z16` horizon curves, `z1` removal sensitivity, and the profiler-free arm of `z13` (Fig. 8) |
| `CV19` | `resample_diff.yml`, `resample_diff_copper.yml` | **manuscript source** | `run_paths.PROFILER_ROOT`. Appendix A table (`z6` variant `profiler`), the profiler-bearing arm of `z13` (Fig. 8), and the differential arm of `z7` (Fig. 5) |
| `CV16stateless` | `resample_stateless.yml` | **manuscript source** | "No state in, absolute out" arm of `z7_StructureCompare.py:49` (Fig. 5) |
| `CV18_raw` | `resample_raw.yml` | **manuscript source** | "State in, absolute out" arm of `z7_StructureCompare.py:50` (Fig. 5) |
| `CV31` | `resample_cv31.yml` | active | Profiler x target-structure design, cell: profiler + differential target. State offered as a candidate predictor; feature selection decides |
| `CV32` | `resample_cv32.yml` | active | Cell: no profiler + differential target. Evaluates on the full held-out record, so it leads the run order |
| `CV33` | `resample_cv33.yml` | active | Cell: profiler + absolute target |
| `CV34` | `resample_cv34.yml` | active | Cell: no profiler + absolute target |
| `CV25` | `resample_cv25_profiler_state.yml` | **superseded** | Profiler x state design, cell: profiler + state. Sweep partial, 9/14. Superseded by CV31: the state dimension is a feature-selection question, and these samples predate the MC replicate seeding fix (`7a4e3ea`), the state-aggregation fix, the perturbation clip and the excluded turbidity calibration record. Retained as the only record of the pre-fix behaviour |
| `CV26` | `resample_cv26_profiler_stateless.yml` | **superseded** | Cell: profiler, stateless. Never run. The stateless dimension no longer exists |
| `CV27` | `resample_cv27_profilerless_state.yml` | **superseded** | Cell: no profiler + state. Sweep complete, 14/14. Superseded by CV32 for the same reasons as CV25. The most complete pre-fix run and the reference for what changed |
| `CV28` | `resample_cv28_profilerless_stateless.yml` | **superseded** | Cell: no profiler, stateless. Never run beyond one stray target |
| `CV29` | **generator missing** | **superseded** | Was a byte-for-byte copy of CV25 restaged after the `_state` aggregation fix. Never run; replaced by CV31, which is generated rather than copied |
| `CV30` | **generator missing** | **superseded** | Was a byte-for-byte copy of CV27. Never run; replaced by CV32 |
| `CV20_profilerless` | **no surviving config** (see gaps) | **hold** | Superseded as a reporting basis by CV22_profilerless (`ASSUMPTIONS_AND_OUTSTANDING.md:112`), but still the hard default of `z14_SelectionStability.py:29`. Held for comparison until a superseding run completes |
| `CV23_profiler` | `resample_diff_profiler_cv23.yml` | **hold** | Smoke run, 2/14 targets. Invariant-check evidence cited at `ASSUMPTIONS_AND_OUTSTANDING.md:515`. Held until superseded |
| `CV24_profilerless` | `resample_diff_profilerless_cv24.yml` | **hold** | 1/14 targets. The verification pair for the `mc_replicates/` check at `ASSUMPTIONS_AND_OUTSTANDING.md:975`. Held until superseded |

## Spikes

Alternative solution methods that were not successful, are not described in the
manuscript, and are **not kept up to date with other project changes**. Retained for
possible future development; do not assume they run against the current code. They are
excluded from any task unless named explicitly — see `CLAUDE.md`.

### Spike model variants

These produce configs and run directories inside otherwise-current roots, so they are easy
to mistake for supported families. They are not.

| Variant | Trained by | Config emitted as |
| --- | --- | --- |
| Recurrent transformer | `src/e_TrainRecurrent.py` | `config_recurrent_transformer_01.yml` |
| LSTM | `src/e_TrainRecurrent.py` | `config_lstm_01.yml` |

The supported families are the Gaussian process, multiple linear regression, XGBoost and the
transformer. `d_RunResample.py` emits the spike configs beside the supported ones with no
marking, which is why this table exists.

### Spike series

| Series | Produced by | Notes |
| --- | --- | --- |
| `pf_ml` | `n_ParticleFilter.py --run-name pf_ml` | Particle filter, ML proposal |
| `pf_mlr` | `n_ParticleFilter.py --run-name pf_mlr` | Particle filter, MLR proposal |
| `pf_mlr_single` | `n_ParticleFilter.py --run-name pf_mlr_single` | Particle filter, single-target MLR. Includes an `mlr_cache/` |

Post-processed by `z3_PFPostProcess.py`. Its `pf_comparison_*.png` figures need a CV
baseline via `--cv-dir`; the original baseline was CV14, which has been disposed of
(below), so those figures now require a retained root and will not be directly
comparable with the originals.

## Supporting directories

These are not results series.

| Directory | Role |
| --- | --- |
| `sensors` | Sensor data-description figures (`b_ExploreData.py`); also the default `normalization.json` source for `e_Train.py:515` and `f_Evaluate.py:79` |
| `calibration` | Sensor calibration and drift analysis (`c_ProcessCalibrationLogs.py`); read by `config_utils.py:36-37` |
| `comparisons` | Write target only: `z7_StructureCompare.py` writes `all/structure_r2.png` here |
| `regression` | Consolidated `.csv` **inputs**, not results. Also the default `--data-root` of `g_`, `h_`, `i_`, `j_`, `k_`, `l_`, `m_`, `z1`, `z2` |
| `classification` | Legacy binarized consolidated inputs (`a_ConsolidateDatasets.py:228`) |

## Disposed 2026-09-23

Removed after confirming that none is a source for anything currently in the manuscript,
and that each one's replacement is complete. `data/output` is gitignored, so these are
not recoverable.

| Path | GB | Files | Released because |
| --- | --- | --- | --- |
| `CV14` | 8.944 | 279,461 | Not referenced by the manuscript in any commit. Its 6.09 GB horizon sweep was formally withdrawn from the draft as "not comparable with the rest of the Results". Produced by `resample_config.yml`, which is retained |
| `CV19_superseded` | 2.056 | 92,897 | The archived predecessor of CV19, created by the `mv` convention at `RUN_OUTPUT_MANIFEST.md:80`. CV19 holds the same 14 targets, all with complete final metrics |
| `CV15profileless` | 1.834 | 75,205 | Never reported. The profiler-free line was superseded by CV20_profilerless and then CV22_profilerless, which is complete. Produced by `resample_profileless.yml`, which is retained |
| `CV22_archive` | 1.128 | 43,392 | Pre-rerun archive of 3 targets (Chromium, Lead, Total coliforms), no `summaries/`, referenced nowhere. All 3 are present with complete final metrics in CV22_profilerless |
| `CV17_recurrent` | 0.639 | 12,957 | Staged and never analysed: no `summaries/`, zero sweep results. Produced by `resample_recurrent.yml`, which is retained |
| `pipeline_logs` | 0.096 | 69 | April 2026 run logs |
| `SMOKE_RUN` | 0.062 | 1,173 | Scratch |
| `SMOKE_GP` | 0.004 | 115 | Scratch |
| `regression/Consolidated_sparse.csv.bak` | 0.010 | 1 | April backup; the live file and a `.pre_gapfill` checkpoint both date from 28 August |

**Recovered: 14.773 GB, 505,270 files.** `data/output` went from 43.18 GB / 1,241,014
files to 28.40 GB / 735,744 files.

The splitting configs for the disposed series are deliberately kept — they are the record
of how those runs were made, and are cheap.

## Known provenance gaps

1. **CV20_profilerless has no surviving config.** No file in `data/input/splitting/`
   writes to it; it was most likely produced by an earlier revision of
   `resample_diff_profilerless.yml` before that file was repointed at CV22_profilerless.
   Its settings are not reconstructible from the repo.
2. **CV29 and CV30 named a generator that does not exist.** Resolved by retiring them: both were copies rather than generated roots, neither was ever run, and CV31 and CV32 replace them with series generated from a recorded config.

3. **`resample_SCADA.yml:11` targets `../../output/SCADA`**, a directory that has never
   existed. Either the run was never made or its output was discarded untracked.
4. **Ten scripts default `--data-root` to `data/output/regression`**, which holds
   consolidated inputs and no run tree. `z1_FeaturePostProcess.py --figures-only` with no
   root argument therefore points at a directory with no `summaries/`. Pass an explicit
   `--data-root`.
