# Assumptions and Outstanding Content

Status as of the CV19 re-run. Anything listed under *Blocked on the re-run* has been
implemented in code and verified on a single-target smoke test, but its numbers cannot be
quoted until the full sweep completes.

## Resolved

- The common predictor pool contains 17 external Surface, SCADA, and weather series. A
  configuration with 18 input columns holds those 17 plus the target-specific state feature;
  it does not hold 18 external predictors.
- **Window lengths are 167 and 671 rows**, not 168 and 672. The split across targets was
  correct; the counts were off by one.
- **Every method is scored on one evaluation set per target**, obtained by intersecting the
  held-out samples of the compared configurations (`z8_CommonSetMetrics.py`). This was
  necessary rather than cosmetic: methods differ in how many samples they forfeit to
  incomplete predictors, and an `R^2` computed on 5 samples is not comparable with one
  computed on 22. Leakage cannot occur, because a sample enters the shared set only if every
  compared method held it out; the script asserts this rather than assuming it.
- **NRMSE is `RMSE / sigma_record`**, where `sigma_record` is the target standard deviation
  over the complete record. Normalizing by the evaluation-set standard deviation instead
  would make NRMSE exactly `sqrt(1 - R^2)`, restating a number already reported. With the
  evaluation-set scale taken about its own mean and without a degrees-of-freedom correction,
  the relation in Section 2.6 holds to floating-point precision.
- **One sigma per target.** Previously `std_target` was recomputed per configuration over that
  configuration's surviving samples, taking 4-9 distinct values within a single target and
  differing between a model and its own reference by up to a factor of three. Enforced at the
  final assembly point in `z1`, after the MLR rows are merged, which is where an earlier fix
  was being silently undone.
- **The differential target is differenced against the window-start value** for all 14 targets,
  verified by `v1_CheckTargetDefinition.py`, which inverts both columns through
  `normalization.json` before comparing them.
- **Predictions are constrained to the target's normalized support.** A Gaussian process with
  an unbounded linear kernel term returned values up to 106x outside `[0, 1]`, and one such
  point is enough to dominate a squared-error metric. Every clip is counted and reported.
- Statistical support is reported as a four-level verdict (supported / directional /
  underpowered / not supported), not the retired 0--5 evidence score.

## Resolved, and it changed the results

- **The structure comparison was mislabelled.** CV16stateless and CV18_raw carry byte-identical
  target values, and both differ from CV19: neither targets a difference. Section 3.1's middle
  structure is "no state in, **absolute** out", and no configuration combining a differential
  target with no state input was ever run, so that cell of the design is empty. Because an
  absolute series is far more autocorrelated than its differences, `R^2` is not directly
  comparable between the differential and absolute structures.
- **The Gaussian process was never fitting its kernel.** The flattened predictor window gave it
  835 input dimensions and 835 ARD lengthscales to estimate from as few as 11 training samples;
  cross-validated early stopping halted after a median of 24 of 250 configured epochs, leaving
  every hyperparameter at its initial value. Four variants now separate the two causes, and on
  the smoke test the aggregated variants trained to 250 epochs with genuinely differentiated
  lengthscales. Section 2.3 has been corrected accordingly.
- **The differential target was quantized by a fixed `round(3)`.** Harmless in ug/L or CFU;
  catastrophic for filtered Copper in mg/L, whose entire range is 0.004, leaving a five-level
  target with 75% of values exactly zero. Copper filtered is the only target whose data changes.

## Blocked on the re-run

- Sections 3.1 (counts), 3.3, 4 and 5, and Figures 6 and 8. Every number in them must be
  re-derived from the common evaluation set; the number-independent corrections are already in.
- On the pre-re-run tree the common-set analysis already moves the headline: a learned model is
  the best predictor for 6 of 14 targets rather than 8, and **no target reaches "supported"**.
  Those figures will move again once the Gaussian process actually fits.
- Whether a dedicated profiler-free arm is still needed. Aggregation both cuts the input
  dimension and relaxes the missing-value rule, nearly doubling GP coverage on the smoke test
  (5 to 9 of 20 segments), but that run's budget retained only profiler-containing subsets.
  Decide from the real run's coverage.

## Known limitations to state, not fix

- **The reported accuracy is a maximum over many scorings of the same test split.** The
  predictor search retains several subsets and every subset-by-method combination is scored on
  the holdout, with the best reported. This is selection on the holdout, not training-data
  leakage, but it is not an unbiased estimate. The number of configurations behind each result
  is now reported. Selecting on a training-split criterion instead would need per-configuration
  cross-validation scores, which are not currently recorded.
- Several targets yield very few independent holdout measurements, and at `n = 5` the smallest
  attainable two-sided p-value is 0.0625, so no outcome can reach significance. This is why the
  verdict scheme separates *underpowered* from *not supported*.
- Models are trained on configuration-specific splits, so the common-set comparison is fair
  between the selected models but not between identically trained ones.

## Content requiring confirmation before submission

- The author-contribution statement and acknowledgments, with all co-authors.
- The served population, raw-water abstraction depth, regulatory context, and described
  catchment hazards, against project records.
- That the study period and all figure data correspond to the final analysis package.
- Whether the supplementary material should carry the full architecture defaults,
  hyperparameter-search spaces, and per-target performance tables, which are compressed in the
  main manuscript for length.

## Abstract and Introduction (revised separately)

Two statements no longer match the results and are left for the author:

- The abstract describes "a multi-test evidence score"; that machinery is now a four-level
  verdict.
- The abstract states that "several targets that showed positive relative skill were not
  predictable in absolute terms". Re-check against the final results: on the pre-re-run common
  set every target had `R^2 > 0` for at least one method.
- The Introduction refers to "simple target-specific baseline models"; the reference set
  includes a tuned multiple linear regression in three aggregation variants, which is not
  simple in the same sense.

## Reporting basis moved to CV22_profilerless

`utils/run_paths.REPORTING_ROOT` now points at `data/output/CV22_profilerless`. Every
z-script that defaults its root follows it, so the manuscript table, the method
comparison, the retention counts and the profiler contrast are all built from that arm.
`data/output/CV19` remains the profiler-bearing arm behind Appendix A and Section 3.2,
and `data/output/CV20_profilerless` is superseded and no longer feeds the paper.

## MLR reclassified as a predictor-driven method

MLR is counted among the machine-learning methods and is no longer in the skill
denominator, which is now `{naive, seasonal, linear}` alone. The distinction the paper
draws is whether a method reads the predictors, not whether it is classical or recent.
Two consequences, both already reflected in the text:

- The three aggregation variants are now one family with three configurations, scored
  like the four GP kernels or the three XGBoost variants. `MLR-12` and `MLR-All` no
  longer exist as separate columns in `common_set_metrics.csv`.
- Skill is a weaker bar than it was, because the reference set no longer contains a
  fitted regression on the predictors. Every one of the 14 skill scores is positive, and
  Section 4 states explicitly what that does and does not establish.

## Horizon sweep: two defects found and fixed

The sweep was run on `CV22_profilerless` at 0, 12, 24, 48, 96, 144 and 168 hours with six
replicates. Two defects were found and fixed before the results were used.

1. **GP kernel.** `matern52+linear` was built with gpytorch's `AdditiveKernel`, which
   accumulates terms as linear operators. `LinearKernel` returns a low-rank
   `RootLinearOperator`, so summing it onto the dense Matern term dispatched to
   `add_low_rank`, which takes an SVD of the concatenated root. On E. coli, whose
   differential is mostly zeros, that SVD did not converge and killed the run.
   `utils.gp_utils.DenseAdditiveKernel` now sums the terms densely: identical to 9e-16,
   `add_low_rank` called zero times instead of once, and the failing case trains.

2. **XGBoost tuning cache.** `k_RunHorizonSweep` hardcoded the cache filename as
   `xgb_cv_tuning_cache.json`, but `e_Train` resolves it per window representation
   (`..._stats-6h.json`, `..._stats-24h.json`). The four targets whose winning XGBoost
   uses an aggregated window were handed no cache and **re-tuned hyperparameters at every
   horizon**, which is a different protocol from the ten targets that kept theirs. The
   name is now derived from the winning config, and the sweep warns rather than
   re-tuning silently. Those four targets were deleted and recomputed.

The sweep no longer aborts on a failed replicate: failures are recorded, printed,
collected into `summaries/horizon_failures.csv` and summarized at the end, and a horizon
that loses every replicate is reported as absent rather than plotted as zero.

## The horizon-0 anchor: diagnosed, fixed, now exact

An earlier draft of this file and of Section 3.4 attributed a horizon-0 mismatch on
Cadmium and Turbidity to selection optimism. **That was wrong.** The cause was the
XGBoost training budget, and it is now fixed; all 14 targets reproduce Table 3 at
horizon 0, 13 to four decimal places and pH to within its own replicate spread.

The mechanism: e_Train sets the training budget from an internal cross-validation on the
training split, and every reported model stops on that budget
(`cv_epoch_budget_exhausted`). The estimator only runs when `early_stopping_rounds` is
set, and applying the XGBoost tuning cache overwrote it. Inside the horizon configs the
estimator therefore never fired and the model trained the cache's full `n_estimators`:

| target | reported rounds | horizon-0 rounds (before the fix) |
|---|---|---|
| Turbidity | 4 | 204 |
| Total coliforms | 10 | 77 |
| Cadmium | 20 | 117 |
| Lead | 161 | 282 |
| Copper filtered | 231 | 232 |

Cadmium's R^2 moved from +0.428 to -0.510 on that alone. The two prediction series were
correlated at 0.995 and differed only in amplitude (sd 0.091 against 0.056, target sd
0.055), which is the signature of over-boosting and not of a different fit. Copper
filtered reproduced before the fix only because 231 and 232 rounds are the same model --
not, as first claimed here, because it has no selection optimism.

Inlining the tuned hyperparameters and switching the cache off restores the estimator.
It then re-derives the budget from each horizon's own training data, so horizon 0
reproduces the reported model and longer horizons get the budget their own data supports
(Cadmium: 20 rounds at horizon 0, 1 round at 168). Because that estimate uses only the
training split, it introduces no comparison the horizon is not entitled to make.

## Family audit of the horizon sweep

Checked for the same class of defect:

- **GP: clean, verified.** All nine GP targets reproduce their reported epoch budget
  exactly at horizon 0 (1, 55, 56, 71, 110, 120, 135, 250, 250) and adapt beyond it. GP
  has no tuning cache, so nothing ever suppressed `_gp_cv_estimate_epochs`. **XGBoost was
  the only family with a cache, which is why it was the only family affected.**
- **Transformer:** same CV-epoch mechanism, no cache, structurally immune. Never
  exercised here -- it is not the best method for any target.
- **MLR:** deterministic, collapsed to one replicate, no budget estimator. Also never
  exercised.
- **Prediction clipping:** the sweep calls `f_Evaluate`, so `_clip_to_target_support`
  applies; all predictions checked sit inside the normalisation support. Lead's R^2 of
  -9.5 is a genuine score on a low-variance evaluation set, not an unclipped blowup.

## Replicates do exactly nothing on this predictor set, and why

13 of 14 targets produce **bit-identical** predictions across all six replicates -- equal
MD5 of the float64 prediction vectors, not merely equal to rounding. pH is the only
exception (6 distinct vectors, sd 0.011 at horizon 0).

Two independent causes, one per family:

1. **The Monte Carlo perturbation is inert.** All 10 MC replicates of a segment are
   byte-identical. Uncertainty perturbation only acts on predictors carrying a
   calibration uncertainty spec, and on the profiler-free set none of the retained
   predictors do -- the specs exist for the Surface-profiler channels, which this arm
   excludes. So there is no draw for a per-replicate seed to vary, and the GP's
   uncertain-input kernel sees zero input variance.

2. **XGBoost's `random_state` never reaches the model.** `k_RunHorizonSweep` writes it
   per replicate ([0..5], confirmed in the configs), but `e_Train` builds the XGBoost
   constructor arguments from an explicit key list -- `tree_method, objective,
   n_estimators, max_depth, subsample, colsample_bytree, min_child_weight, gamma,
   reg_lambda, reg_alpha, learning_rate, n_jobs` -- with no `random_state`. The name
   appears nowhere else in `e_Train` except a default value and the train/test split
   seed. XGBoost therefore uses its own default of 0 on every replicate, even though
   `subsample` (0.51-0.86) and `colsample_bytree` (0.40-0.86) are active and would
   otherwise make the seed matter. **Not fixed** -- fixing it would shift the reported
   means for the five XGBoost targets and require re-running them.

GP has no other fitting randomness by construction: exact GP, full-batch Adam, fixed
initialisation, no dropout or minibatching. Its `uncertain_kernel_mc_seed` *is* connected
(read in `e_Train`, passed to `build_base_kernel`) and pH proves the path is live.

### `--mc-seed-per-replicate`

Added to `k_RunHorizonSweep`, off by default. Each replicate resamples with
`random_seed + rep_idx`, so the replicate spread measures propagated predictor
measurement uncertainty. Intended for a profiler-bearing run, where the perturbation is
live. Default-off because it moves resampling from once-per-horizon to once-per-replicate
and resampling is the dominant cost of the sweep.

Verified on Turbidity across all 7 horizons: 42 resamples fired and every horizon mean
was identical to the shared-resample run to 1e-9, which is the correct outcome for an
inert perturbation. **This verifies the plumbing, not the effect** -- whether the spread
is informative can only be established on a run where the perturbation perturbs.

A guard was added with it: `_perturbation_is_inert` hashes the MC copies of a segment
after each resample and warns when they are identical. On a profiler-bearing run, that
warning *not* firing is the confirmation the perturbation is live.

### Runtime

Replicates cost about 30 min of the ~3.5 h sweep (14%), measured at 3.3-4.2 s per
replicate. An earlier note in this file claimed `--replicates 1` would be "six times
cheaper"; that was wrong -- replicates are six times the *training*, and training is a
minority of the sweep. Resampling dominates and is intrinsic to varying `gap_rows`.

### XGBoost trains on duplicated rows

Because the MC replicates are identical copies, the XGBoost path trains on 10x duplicated
data: Cadmium 520 files from 52 distinct segments, Total coliforms 1150 from 115. The GP
path does not (Nickel 52/52, pH 30/30). This is a modelling caveat rather than a runtime
one: `min_child_weight: 6`, tuned on 10x-duplicated data, is effectively `0.6` relative
to the distinct sample count, and `subsample`'s effective draw size is likewise inflated.
Worth stating wherever those hyperparameters are quoted.

A hypothesis that was checked and **rejected**: that the duplication made the GP Gram
matrix singular and thereby caused the `A not p.d.` warnings and the non-convergent SVD
fixed earlier in this work. GP trains on distinct segments only, so it cannot be the
cause; the `DenseAdditiveKernel` fix rests on its own evidence.

## Seed variance and the GP candidate pool: what changed in Table 3

Two defects in the sweep are large enough to move reported results. Both were measured
incrementally rather than by repeating the 23.7 h sweep.

### 1. XGBoost was fitted at one unrecorded seed

`hyperparameters.random_state` was absent from the XGBoost constructor arguments in
`_train_xgb_model`, so every candidate was fitted at XGBoost's default of 0 whatever the
config said, and each winner was chosen from a single draw. With the seed honoured, six
seeds give an R^2 standard deviation of **median 0.031, p90 0.138, max 1.045** across 216
reproducing candidates. `xgb_03` is the most volatile variant (median 0.046).

Re-selecting each target on the six-seed mean (`v3_SeedVarianceRefit.py`, margin 0.45,
233 candidates, ~1.7 h):

| target | reported | seed-averaged | winner |
|---|---|---|---|
| Cadmium | +0.4285 | **+0.2787** | XGB (kept) |
| Total coliforms | +0.2246 | **+0.1411** | **GP** (was XGB) |
| Turbidity | +0.2874 | **+0.2405** | **GP** (was XGB) |
| Copper filtered | +0.2925 | +0.2939 | XGB (kept) |
| Lead | -0.0001 | -0.0057 | XGB (approx. 0 either way) |
| 8 others incl. pH | unchanged | unchanged | GP/MLR |

All 13 targets pass the margin check at 0.45. The result is corroborated by an
independent path: the horizon sweep's horizon-0 rows are a six-seed refit of each
target's winner, and agree with `v3` **exactly** on Cadmium (+0.2787) and Copper
(+0.2939).

### 2. The GP candidate pool was truncated by a kernel crash

47 candidate GP configurations produced only `train_files.txt`/`test_files.txt`, having
died at a *logging-only* forward pass. 46 are the two `+linear` variants; `gp_04`
(`matern52` alone) failed zero times. `DenseAdditiveKernel` removes the failing path.
Re-running them (`v2_RecoverFailedGPConfigs.py`) recovers two real improvements:

- **Colony count** +0.3830 -> **+0.4798** (`gp_01`)
- **Arsenic** +0.3841 -> **+0.4284** (`gp_01`)

The rest score below their target's reported best, several catastrophically (Lead to
-52.9). So the crashes were not selecting against bad configurations; they removed good
and bad indiscriminately.

### So five of fourteen targets have a reported best that does not survive

Arsenic and Colony count gain a better GP; Turbidity and Total coliforms lose XGBoost to
GP; Cadmium keeps XGBoost at a much lower value. **Table 3, Figure 12 and the Section 3.3
narrative all need revising before submission**, and the horizon sweep takes its winners
from Table 3, so it must be re-run after any change to the selection.

### Residual uncertainty, not yet closed

- **17 of 233 candidates do not reproduce at seed 0**, all `xgb_01` (flattened window),
  across Colony count, Copper, E. coli, Total coliforms and pH. They are excluded from the
  re-selection rather than trusted. All scored below their target's best when originally
  fitted, so it is unlikely any would win, but that is not proven.
- **8 of the 47 GP configurations were never re-run**: one hung indefinitely (below) and
  took the remainder of the queue with it.
- **The beam-search trajectory cannot be recovered incrementally.** Which subsets were
  explored is fixed at what the sweep found, and both the GP crashes and the surrogate's
  seed would have changed it. Only a full re-run addresses that.

## Non-reproducibility and hangs

- **Transformer fits were not reproducible at all.** Nothing called `torch.manual_seed`
  and the transformer never read `random_state`, so weight initialisation and dropout came
  from PyTorch's unseeded global generator. Two fits of one unchanged config gave R^2
  +0.4797 and +0.4326. `_seed_model_rng` now seeds `random`, `numpy`, `torch` and CUDA and
  sets `cudnn.deterministic`; the gap falls to 0.0057 but is **not** zero, because CUDA
  reductions remain nondeterministic without `use_deterministic_algorithms(True)`. All 294
  Transformer fits in CV22 are unreproducible single draws. The Transformer wins no
  target, so no reported result depends on one, but it did compete in the selection.
- **One GP configuration hangs indefinitely**: Total coliforms `gp_01_r167_f11`
  (flattened, ~1837 input dimensions) ran 25 minutes on 0.016 s of CPU. `v2` and `v3` now
  pass `timeout=1800` to every subprocess. `h_RunMCFeatureSelectionSweep` trains
  in-process and has no equivalent guard; a hang there still stalls a sweep silently.

## Pipeline readiness for a full re-run

In place, and inherited by any future sweep because they live in `e_Train`/`gp_utils`:

- `DenseAdditiveKernel` -- removes the GP crash, so the beam trajectory is no longer
  truncated
- XGBoost `random_state` reaches the model
- `_seed_model_rng` seeds the transformer and GP
- `--seeds N` / `--seed-base` on `h_RunMCFeatureSelectionSweep`: fits each candidate N
  times and selects on the mean, for XGBoost and the transformer only, since GP and MLR
  are deterministic. **Default 1 reproduces the previous behaviour exactly.** Implemented
  but **not exercised end to end**, because testing it means running the sweep.

Horizon-sweep fixes (tuning-cache filename, CV round budget, truncated-history baselines,
`--mc-seed-per-replicate`, failure recording) are described in their own sections above.

## Table 3 and the horizon curve disagreed by construction, not by error

Both numbers were correct for what they computed; they computed different things.

The horizon sweep fits a target's winning configuration six times at each horizon, differing
in **nothing but the model seed** -- verified on Cadmium's horizon-0 replicates, which share
one predictor set, one aggregation (`stats:6h`), one round budget (20) and one resampled
`samples/` directory, and differ only in `random_state` 0-5. `z2`/`z16` then plotted the
**mean of the six R^2 values**. Table 3 reports the **R^2 of the mean prediction**, the
six-seed ensemble `z17_ApplySeedEnsembles` installed. Squared error is convex in the
prediction, so by Jensen's inequality these cannot be equal:

    R^2(mean prediction) - mean(R^2) = n * (mean across-seed variance) / SS_tot >= 0

Measured on Cadmium at horizon 0 the two sides agree to six decimals (0.035342): per-seed
values +0.4285 +0.2035 +0.2331 +0.2768 +0.2738 +0.2566, mean of the R^2s +0.278715, R^2 of
the mean prediction +0.314058, Table 3 +0.314058. The figure's leftmost point therefore
disagreed with the results table for exactly the targets whose winner is stochastic, and
matched for the deterministic ones because their replicates are identical and the two
aggregations coincide.

**Ensembling is the side converged on.** Only the ensemble has a prediction series behind it,
so R^2, the skill score and the permutation test all describe one model; and a six-seed
ensemble is deployable, where "the average score of six models you would still have to choose
between" is not.

`z18_HorizonEnsembles.py` writes `lookahead_ensemble.csv` per target by averaging the
replicate prediction vectors already on disk -- **nothing is retrained** -- and
`z16_HorizonCurves.py` prefers it, falling back to the replicate mean with a warning.

### Residual mismatches after the fix: 11 of 14 exact, 3 explained

| target | Table 3 | horizon 0 | difference | cause |
|---|---|---|---|---|
| pH | +0.5348 | +0.5207 | -0.0141 | GP `uncertain_kernel_mc_seed` varies 0-5 across replicates; the reported fit holds it at 0 |
| Turbidity | +0.2405 | +0.2374 | -0.0031 | same |
| Lead | +0.0216 | -0.0006 | -0.0223 | signal below its own seed noise (below) |

The GP is deterministic in its optimizer but **not** in its uncertain-input kernel, which
draws 64 Monte Carlo samples; the horizon sweep varies that draw per replicate. Prediction
vectors still correlate at 0.99988 (pH) and 0.99999 (Turbidity), with maximum per-point
differences of 0.004 and 0.0016. This is a real inconsistency in what the two paths hold
fixed, small enough to disclose rather than re-run.

**Lead is a different matter and is now reported as such.** Its zero-hour ensemble prediction
has a standard deviation of 0.00192 against a target standard deviation of 0.02255, while the
spread across its own six seeds is 0.00194 -- *larger than the signal*. The configuration
returns what is effectively a constant forecast whose variation is entirely seed noise, which
is why two six-seed ensembles of the same run id disagree by 0.022 and why their prediction
vectors correlate at 0.00000 despite differing by at most 0.0086. Section 3.4 states this.

## `--seeds` was broken in three ways, and is now consolidated into the main pass

The flag was added to make selection seed-robust. Examined properly, it was doing close to
the opposite. Three defects, all fixed in `h_RunMCFeatureSelectionSweep.py`.

### 1. Seed replicates were competing as candidates

`z8_CommonSetMetrics.load_runs` scans `feature_sweeps/` indiscriminately: every directory
with a recognised family prefix and a `predictions.csv` becomes a candidate. The per-seed
replicate runs landed there, so `--seeds N` added N single-seed draws of every stochastic
candidate to the pool and let `z8` select the maximum over all of them. That is a
winner's-curse amplifier -- the flag made selection *more* seed-fragile, not less. In the
CV23 smoke run, **41 of 119 scoreable directories were replicates**.

### 2. The seed suffix collided with an existing subset label

Replicates were suffixed `_s%02d`. But `_s01` is already a subset label in the
run-directory convention, beside `_k01`-`_k04`, `_l01` and `_m01`: **CV22 contains 182
legitimate `_s01` directories**. A seed replicate and a subset run were therefore
indistinguishable by name, and could collide outright. The CV23 smoke run's 27 `_s01`
directories were 13 real subset runs plus 14 seed replicates.

Replicates are now suffixed `_seedNN`, which matches no subset pattern, and `z8` excludes
them by that suffix. They stay inside `feature_sweeps/` because the two resolvers disagree
about what a forecast name is relative to -- `e_Train` joins it under `forecasts/` verbatim
while `h_` prepends the sweep namespace -- so a directory prefix puts the replicate in
`forecasts/feature_sweeps/seed_reps/` for one and `forecasts/seed_reps/` for the other.
That mismatch is what a first attempt at relocating them hit.

### 3. The reported result was still a single draw

Seed averaging stopped at the beam search. The final per-family re-fit -- the run `z8`
scores and the results table quotes -- was one fit. So the reported number was still chosen
from a single draw of a model whose seed moves R^2 by a standard deviation of 0.03 at the
median and up to 0.44.

That stage now installs the **mean prediction vector** across seeds. Averaging predictions
rather than scores is the only internally consistent option: there is no prediction series
whose R^2 is the mean of six others, so a mean R^2 reported beside a significance verdict
computed from one seed would quote two different models. The ensemble has one prediction
vector, so R^2, the skill score and the permutation test all describe the same object, and
it is deployable where "the average score of six models you would still have to choose
between" is not.

An earlier attempt was reverted for the wrong reason and the claim is withdrawn: the note
that this "cannot be seed-averaged without restructuring" was too pessimistic. The first
hook was simply placed before `evaluate_single_config` had written any predictions, and
trained 27 extra runs only to discard them. The correct hook is immediately after that call.

### The GP is not deterministic, and was being treated as though it were

`v3` measured the GP's seed spread as exactly 0.0000 and concluded it was deterministic. Its
optimizer is, but its uncertain-input kernel draws 64 Monte Carlo samples per fit from
`uncertain_kernel_mc_seed`, which `v3` never varied. This is the direct cause of the
remaining Table 3 / horizon-curve mismatches on pH (0.014) and Turbidity (0.003), whose
horizon replicates *did* vary it. `_is_stochastic_model` now returns True for a GP with the
uncertain kernel enabled, and `_seeded_variant_config` sets that seed alongside
`random_state`.

### Consequence for `z17`

`z17_ApplySeedEnsembles` is now a repair tool for CV22 only, which was fitted before any of
this existed. New runs need no post-processing step: `--seeds N` produces the ensemble
in-pass, and `feature_sweep_final_metrics.csv` is recomputed from it rather than left
holding the single-seed score.

That staleness is real in the current CV22 tree and worth recording: `_pooled_r2` recomputed
from `predictions.csv` reproduces the reported `r2` for **all 318** untouched runs and for
**none of the 80** that `z17` overwrote -- a clean 2 x 2. Table 3 is correct, because `z8`
reads `predictions.csv`; `h_`'s own metrics file is not.

## Verification and provenance, added because assertion kept failing

Sorting the retractions in this record by how the claim was arrived at gives one clean
split: **every claim that had to be withdrawn was asserted from reading code or from
plausible reasoning, and every claim that survived was measured first.** A secondary
pattern compounded it -- generalising a timing or a rate from a single target. Two tools
exist to move the pipeline's guarantees out of judgement and into commands.

### `src/v4_CheckPipelineInvariants.py`

Encodes each consistency check that had previously been run by hand, once, and re-found
by hand on the next tree. It imports the pipeline's own `_pooled_r2` and `z8.load_runs`
rather than reimplementing them: a checker carrying its own copy of the metric can agree
with itself while disagreeing with the pipeline, which is the failure it exists to catch.
Non-zero exit on any FAIL, so it can gate a re-run.

| check | catches |
|---|---|
| run integrity | fits that failed silently (the 55 GP configs) |
| metrics vs on-disk predictions | a table describing predictions it no longer matches |
| candidate pool | seed replicates competing as independent candidates |
| horizon anchor | the mean-of-R^2 vs R^2-of-mean mismatch, and a missing `z18` run |
| search discrimination | a surrogate that ranks every subset identically |
| ensemble provenance | an N-seed row with no single-seed predictions kept |
| output containment | summaries written from a different root |

Results on the trees as they stand:

    CV22_profilerless   3 pass, 2 warn, 2 FAIL
    CV23_profiler       6 pass, 0 warn, 1 FAIL
    CV19                                1 FAIL (exit 1)

CV22's two failures are both already-known and already-recorded: the 8 unrecovered GP
configs, and Chromium's degenerate surrogate. Its warnings are the 216 rows `z17` left
stale and the three horizon-anchor residuals (Lead 0.0223, pH 0.0141, Turbidity 0.0031),
all documented above. **Nothing new or unexplained appeared**, which is the useful result:
CV22 does not need re-running on suspicion.

The check also surfaced a figure not previously measured: **Lead's search had 185 of 240
candidates degenerate**, and Color 57 of 232. Lead is the target whose prediction spread
is smaller than its own seed spread, so the two findings agree.

### `src/utils/provenance.py`

Writes a manifest to `<root>/summaries/run_manifests/` before the work starts -- so a
crashed run still records what it attempted -- and stamps it with status and duration on
the way out. Captures argv, all resolved arguments, git commit and **dirty flag**,
package versions, host and timing.

It exists because nothing recorded a run's settings, so "what did CV22 do?" had to be
answered by reading directory names against current defaults, and the defaults had moved.
That produced two wrong answers: whether the reported run used seed replication, and the
fact that the surrogate is chosen among XGBoost configs only. Both are now single fields
in the manifest (`candidate_seeds`, `surrogate_scope`).

Manifests accumulate rather than overwrite: a tree is usually built by several
invocations, and which ones touched it is exactly what gets forgotten.

## How each model family's training budget is actually set

The Gaussian process budget was already known to be unidentified for 93% of final-stage
fits. The same question was put to the other two families, and the answers differ from each
other and from what the code appeared to say.

**The XGBoost results on disk were produced by code that no longer runs, and their
round budgets are pathological.** Every one of the 3381 stored XGBoost runs -- 294
final-stage and 3087 search fits -- records a stop reason of the form "CV-derived budget
exhausted (N rounds from internal CV estimate)", which is emitted only by
`_xgb_cv_estimate_n_estimators`. Their recorded hyperparameters are Optuna-tuned values,
not the configs' round base values, so those fits both tuned and then took the budget from
the CV estimator.

The current code cannot do both. Every final-stage config sets `cv_tuning.enabled: true`,
which routes through `_train_xgb_model_cv_tuned`; that function nulls
`early_stopping_rounds`, passes `disable_early_stopping=True` and passes a non-None
`eval_set_override`, and each of those three independently fails the gate guarding the
estimator. Verified by instrumenting the estimator with a call marker: it fires with
`cv_tuning` disabled and does not fire with it enabled. So the estimator is unreachable
now, and the budget today would be the tuned `n_estimators` instead.

**The stored XGBoost results are therefore not reproducible from their own configs.**
Re-running Cadmium's reported config unchanged gives 188 rounds and a pooled R2 of -0.8054,
against the stored 20 rounds and +0.3141. The rebuilt-arm and unchanged-config runs agree
exactly with each other, so this is not an artifact of the reconstruction. The metric is
not in question either: pooled R2 recomputed from each reported run's own predictions.csv
reproduces `xgb_r2` exactly for 12 of 14 targets.

**The budgets those runs used are far too small to be credible.** Across the 3381 fits the
estimator chose a median of 2 rounds; 2317 fits (69%) got ten rounds or fewer and 2751
(81%) got twenty or fewer, with a tenth percentile of 1. Per target, the reported models
were trained for:

    pH                    1 round     R2 +0.3198
    Arsenic               2           +0.1240
    Total coliforms       3           +0.1362
    Intestinal enteroc.   4           +0.0599
    Turbidity             4           +0.2362
    Colony count          7           +0.3690
    Nickel               10           +0.4185
    Chromium             19           -0.0014
    Cadmium              20           +0.3141
    Color               195           +0.2261
    E. coli             209           +0.1240
    Copper filtered     231           +0.2940
    Zinc                238           +0.3079

Nine of the thirteen were trained for twenty boosting rounds or fewer, and pH's +0.3198
comes from a single round. At the tuned learning rates, which run from about 0.005 to 0.03,
a one-round model is the base score plus one shrunken tree -- very nearly a constant
predictor. This is the collapse the estimator's own docstring says was fixed by averaging
the fold curves rather than taking the median of the folds' `best_iteration`; the measured
distribution shows it was not fixed, and is worse than the figures that motivated the
change.

The consequence reaches past the reported table. The same mechanism set the budget for all
3087 search fits, so the backward-elimination beam search for XGBoost was ranking
near-constant models against one another, which is the most likely reason its candidate
rankings were so often inseparable.

**Both mechanisms are leakage-free**, so none of this is a test-contamination problem:
`_xgb_cv_estimate_n_estimators` and `_xgb_tune_hyperparameters_cv` each receive only
`train_samples` and neither references `test_samples` or `X_test`.

**The `best_median_iteration` override never fires under either arrangement**, because
`cv_tuning.use_early_stopping` is false and so no fold produces a `best_iteration`:
`median_best_iteration` is null in all 8645 trial rows across all 42 tuning studies, and no
stored run records that override as its stop reason.

**The recorded stop reason could not have revealed any of this on its own.**
`cv_epoch_budget_exhausted` is reported by both paths -- the tuned path keys it off
`disable_early_stopping`, which is unconditionally true there. Only the stop reason *text*,
which names the round count and the mechanism, distinguishes them.

Establishing exactly when the code changed would require reading the commit history, which
has not been done.

**The XGBoost round budget is unidentified by cross-validation but lands well anyway.**
Sweeping the round budget to 300 over the same 5 grouped training folds, at each target's
reported operating point, puts the one-standard-error plateau at 300 of 300 rounds for 11
of 14 targets and no fewer than 272 of 300 for the rest -- 0 of 14 identified -- with
per-fold argmins as scattered as [33, 294, 6, 8, 41], and an argmin bearing no relation to
the tuned budget. As on the Gaussian process, a flat fold curve is not evidence that the
budget is harmless, so it was tested against the reported metric directly.

**The transformer's estimator is live and reproducible, but barely better determined than
the GP's.** All 294 final-stage fits were re-run to produce the diagnostics that did not
exist when the tree was built, with no failures. The estimator ran on every one, and the
budget is unidentified for **269 of 294 (91.5%)** against the Gaussian process's 93% -- so
the identifiability rate is essentially the same, and an earlier reading of 0.47 taken from
the first seven targets alphabetically understated it.

What differs is the degree, not the rate. The 1-SE plateau covers a median 0.64 of the
scanned range and a maximum of 0.97, and it never covers the whole of it in any of the 294
fits, where the GP's covered the entire scanned range for 326 of 392. Per-target medians run
from 0.03 (E. coli, the one target where the budget is usually identified -- 11 of 21 fits)
to 0.90 (Total coliforms, 0 of 21). Chosen budgets are small but not degenerate: per-target
medians of 3 to 29 epochs over a scanned range of about 30, with a floor of 2.

The budget is also exactly reproducible, which the GP's tiny-budget XGBoost counterpart is
not: repeating an identical configuration four times on each of three targets returned the
same budget every time, with identical per-fold argmins and identical plateau widths --
spread 0 in 12 of 12.

**The transformer's budget does move its score, but for most targets not by more than
its own seed noise.** With `fixed_epoch_budget` making a leakage-free pinned budget
possible, each target's reported transformer configuration was refit at 1, 5, 15, 30 and
100 epochs, plus a control arm that leaves the config alone so the CV estimator chooses
exactly as the reported fit did. 84 fits, no failures.

The 1-epoch arm is excluded from the summary below. One epoch is not a budget anyone would
choose -- the network is essentially untrained and its predictions are wild, reaching
-164.89 on Lead and -15.05 on Color -- so including it inflates every spread for a reason
that carries no methodological content. The remaining arms, 5 to 100 epochs, bracket every
CV-chosen budget, which run from 3 to 37.

Over 5 to 100 epochs the score moves by more than 0.05 on all 14 targets, more than 0.25 on
10 and more than 1.0 on 5. Long budgets overfit hard on several targets: Cadmium falls to
-5.60 at 100 epochs against +0.02 at 5, Copper to -4.27, Turbidity to -1.65. Five of the
fourteen do best at just 5 epochs while the CV estimator picks 22 to 37 for them.

The shortfall of the CV budget against the best budget tried has a median of +0.0965 and a
mean of +0.2985, against the Gaussian process rule's +0.0110 and +0.0233 -- roughly nine
and thirteen times worse. But that comparison must be read against the noise floor, and the
transformer's is large: the control arm differs from the reported score by a median of
0.0920 and a maximum of 2.3928, and lands within 0.05 for only 3 of 14 targets. The median
shortfall is therefore about equal to the median noise floor, and **the shortfall exceeds
that target's own noise floor for only 6 of 14 targets** -- convincingly on Cadmium
(+1.28 against 0.63), Chromium (+0.19 against 0.06), Arsenic (+0.08 against 0.03) and Zinc
(+0.04 against 0.001), marginally on Lead and Intestinal enterococci. On three targets the
control arm beat every fixed budget.

So the honest reading is that the transformer's CV budget is materially worse than the
Gaussian process's on a minority of targets and indistinguishable from noise on the
majority, and that a budget ceiling matters more than the exact choice: the damage comes
from training too long, not from the estimator picking 22 rather than 30.

**The same sweep exposed something independent of budgets: the reported transformer scores
are single-seed draws with large run-to-run variance.** Refitting a target's reported
configuration and letting the CV estimator choose the budget, as the reported fit did,
reproduces the reported score within 0.05 for only 3 of 14 targets, with a median absolute
gap of 0.0920 and a maximum of 2.3928 (Colony count, where a refit gives -2.2165 against a
reported +0.1763). The budget is reproducible -- 12 of 12 repeats returned identical
budgets, argmins and plateau widths -- so this is weight-initialisation and batch-order
variance in the fit itself, not budget instability. It is the strongest available argument
for seed-ensembling the transformer rather than reporting one draw.

**The transformer as run is leakage-free, but a pinned budget had no leakage-free
expression.** `train_transformer` receives `testloader` as its validation loader and
restores `best_model_state` -- the epoch scoring best on that loader -- whenever
`max_epochs_override` is None. So the `threshold` source, and the `cv` source when
estimation fails and falls back to it, would select final weights using the test split.
That never happened in the reported results: all 297 transformer fits on
CV22_profilerless record `checkpoint_restored=0` and `cv_epoch_budget_exhausted`, so the
estimator succeeded every time. But setting `num_epochs` alone does not avoid the restore,
because it changes how long training runs and not which epoch is kept. A new
`fixed_epoch_budget` hyperparameter sets the override directly, so the last epoch's weights
are kept and no test metric is consulted, which both the budget sweep and any reproducible
fixed-budget run require.

**Two reporting fixes followed from these findings.** XGBoost's estimator now routes
through the shared `_choose_cv_epoch`, so where it is reachable its budget carries the same
plateau, per-fold-argmin and `identified` diagnostics as the other two families; the chosen
round is unchanged, verified against the previous computation on 4000 random fold-curve
cases with zero mismatches, the only divergence being that a NaN in one fold now uses the
remaining folds instead of propagating. And the transformer's stop reason now names the
mechanism that set the budget rather than labelling every fixed budget as CV-derived, so it
does not repeat XGBoost's uninformative-stop-reason problem.

**Two measurement errors in this work, both corrected.** The first version of the XGBoost
budget sweep located each target's tuning cache by guessing its filename from
`sample_subdir`. The cache name is derived from the window aggregation by
`aggregation_slug` instead, so the guessed name exists for no target: 8 of 14 fell through
to the unaggregated `xgb_cv_tuning_cache.json` and the other 6 silently received no tuned
hyperparameters at all, which meant those arms were swept at the config's base settings
rather than the reported operating point. The sweep now resolves the path through the
project's own `_resolve_cv_cache_path` and records which cache it used. Separately,
`xgb_curve.py` called `load_and_split_data` without redirecting `forecast_name`, so it
wrote `train_files.txt` and `test_files.txt` into 14 real `feature_sweeps` run directories
and created one that had not existed. All 14 were verified byte-identical against their own
configs -- the split is read from a pinned split directory rather than re-derived -- so
only modification times changed; the created directory was removed and the script now
writes to a scratch forecast name.

**Chromium and Intestinal enterococci have no tuning cache at the resolved path**, so their
reported XGB fits re-tuned in process and their budget is not recoverable from disk.

## Identical XGBoost tuning across window representations is degeneracy, not a bug

Worth recording because the signature looks exactly like a serious defect and cost three
false alarms to settle.

On Chromium the three XGBoost variants -- window representations `none`, `stats:24h` and
`stats:6h` -- produced byte-identical tuning trials (same MD5 across three separately
created Optuna studies) and identical tuned hyperparameters in all three caches, while the
fits themselves demonstrably used different representations: `input_dim` 11, 44 and 44 at
`seq_len` 167, 7 and 28. That is the signature of the bug the cache-filename fix was written
for, where a variant inherits hyperparameters tuned on another representation.

It is not that bug. Nickel, whose XGBoost is not degenerate, tunes each representation to
genuinely different hyperparameters -- `n_estimators` 213, 158 and 126, `max_depth` 2, 3 and
6, `learning_rate` 0.078, 0.174 and 0.029 -- so the aggregation does reach the tuning
objective and the caches are representation-specific as intended.

Chromium's identical studies follow from its degeneracy. Its XGBoost predicts a constant
under every representation (r2 = -0.0014 in all three), so the cross-validation objective is
flat whatever the features look like, and a study with a fixed seed returns the same trial
from the same flat landscape. Identical tuning on a degenerate target is therefore expected;
identical tuning on a target that is NOT degenerate would be the defect.

Two smaller corrections from the same episode. The tuning caches live in
`forecasts/feature_sweeps/`, not `forecasts/`, so a glob at the wrong level reports one cache
where three exist. And the trials CSV is written under the run's `forecast_name`, so a script
that redirects the forecast name will not find it at `forecasts/<model>/xgb_cv_trials.csv`.

## The full pipeline, end to end on one target: what it cost and what it proved

Chromium was run through the complete sweep on the new root with the settings the previous
run recorded, plus the new seed split: `--no-improve-patience 999`, `--fit-timeout 1200`,
`--seeds 1 --final-seeds 6`, `--stop-on-error`, everything else default. It was chosen
because it is one of the two known-degenerate targets and therefore exercises the least
validated additions.

**Cost, measured rather than projected: 72.5 minutes for the target.** The search took 24m
53s for 231 of 240 evaluations at 6.5 s each; the final stage, including six-seed ensembling
of every stochastic winner, took the remaining 47 minutes. An earlier projection of two and
a half hours was pessimistic by more than double, so six-seed ensembling costs far less than
feared -- roughly 17 hours for 14 targets rather than 35, and close to the previous run's
hour-per-target baseline.

**The surrogate degeneracy guard did what it was written for.** All three XGBoost
configurations and `gp_02` predict a constant on the full feature set (r2 = -0.0014 and
-0.0012), so the pool was widened automatically to the remaining seven configurations, four
constant predictors were excluded from the choice, and `gp_03` was selected at r2 = +0.4764.
Under the previous behaviour the search would have been ranked by a degenerate surrogate
that cannot separate feature subsets. The search then improved monotonically from 11
features to 7 (0.4764, 0.4989, 0.5091, 0.5103, 0.5203) and terminated on candidate
exhaustion rather than on patience -- the patience counter reached 3 of 999, which is the
expected silence under the recorded settings, not a defect.

**Every gate passed.** The nine invariant checks: 9 passed, 0 warnings, 0 failures, over 531
run directories -- including 200 seed replicates with none reaching the candidate pool, 70
of 70 metrics rows agreeing with their predictions to 1e-6, and, for the first time, "84
XGBoost runs verifiable, 0 contradicting their config, 0 predating the budget_mechanism
field". CV diagnostics were recorded on all 342 Gaussian process and all 84 transformer
fits. The provenance manifest finalised with `status=completed`, `candidate_seeds=1`,
`final_seeds=6`, and a clean working tree at commit 7e25efc, so the run is reproducible from
its commit alone.

**Reproducibility, tested rather than assumed.** Re-running six stored configurations across
the three learned families reproduced their pooled R2 to within 2e-4 -- 0.0000 for the
Gaussian process and transformer, -0.0002 for XGBoost, which is consistent with GPU and
thread nondeterminism. This is the check that CV22 fails: there, re-running Cadmium's own
configuration gives -0.8054 against a stored +0.3141.

**Comparability on this target: slightly better, same winner.** On the identical 22-segment
evaluation set, the Gaussian process improves from +0.5010 to +0.5203 with the same `gp_03`
variant on a different subset (7 features rather than 4). XGBoost is unchanged at -0.0014
and still degenerate. The transformer falls from +0.0573 to +0.0149, which is well inside
its measured noise floor and does not affect the target's outcome.

The most reassuring number is the one that did not move: MLR (+0.4605), naive (-1.8494),
seasonal (+0.0484) and linear (-0.5653) are **identical to the last decimal** between the two
roots. Those families are deterministic, so identical values across independently staged
trees show the re-staging reproduced the data, the splits and the evaluation exactly, and
that the differences above are model behaviour rather than a changed pipeline.

## Validation of the new root: what 42 fits across all 14 targets established

One fit per target per family on the staged profiler-free root, no feature search: 42 fits,
**zero failures**, every budget mechanism recorded as intended -- `cv_round_estimator` on 14
of 14 XGBoost fits, CV diagnostics on 14 of 14 Gaussian process and 14 of 14 transformer
fits.

Wall times, which is what a timeout should be set from rather than a guess: the Gaussian
process is fast (median 7 s, max 8), the transformer moderate (median 27 s, max 56), and
XGBoost slow because of its Optuna study (median 172 s, max 296). `--fit-timeout 1200`
covers four times the slowest observed fit; CV22 used 1800, which is also safe.

**The transformer keeps its cross-validated budget.** The plan was to replace it with a
constant, justified from the estimator's own clustering, and the pre-registered rule was to
adopt a constant only if the per-target budgets clustered within 25 epochs. They do not:
across the 14 targets the estimator chose 3, 7, 7, 7, 13, 16, 18, 20, 26, 28, 28, 28, 30 and
37 epochs, a spread of 34, with only 8 of 14 inside the interquartile range. A single
constant would be wrong by more than a factor of two at both ends.

Two further reasons not to force it. The measurement is on the FULL feature set, while the
constant would apply to post-search winners, so the clustering that matters cannot be known
until a search has run -- CV22's tighter 22-37 range was measured on winners, and that is a
different population. And the saving is small in absolute terms: the estimator costs about
13 s on a 27 s fit, and the transformer is fitted only at the final stage, roughly 21 times
per target, so dropping it would save a few minutes per target rather than the doubling the
percentage figures suggest. The earlier recommendation to fix the transformer budget is
therefore withdrawn on this tree, on its own pre-registered criterion.

## Preparing the re-run: the seed split, and two tooling traps

**The search and the final stage now take separate seed counts.** `--seeds N` reached every
stochastic candidate in the beam search as well as the reported re-fit, so N=6 multiplied
the search roughly sixfold for the Gaussian process, transformer and XGBoost. `--final-seeds`
governs the final stage alone and defaults to 0, meaning "follow --seeds", so every
invocation written before the split behaves exactly as it did. The intended setting for a
re-run is `--seeds 1 --final-seeds 6`: candidates are ranked on single draws, and only the
winning configuration of each family is refitted at further seeds. Both values are recorded
in the provenance manifest.

**`--dry-run` is neither free nor side-effect-free.** The XGBoost CV tuning and the split
pinning happen during dataset discovery, which runs before the dry-run branch is reached, so
a dry run spends a 250-trial Optuna study per window aggregation and writes
`forecasts/pinned_split` and `xgb_cv_tuning_cache*.json` into the root. Two attempts to use
it as a cheap audit consumed about an hour of CPU between them and had to be killed and
cleaned up. The flag's help text now says so. A static audit of the generated configs
answers the same questions for free, and is what the validation tier uses.

**Configs staged outside their own directory need an absolute `data_dir`.** The generated
configs carry `data_dir: '.'`, resolved against the config file's location, so writing a
modified copy to a scratch path silently points the data directory at the scratch path. Any
script that stages a config elsewhere -- and several of the measurement scripts do -- must
set `data['data_dir']` explicitly.

**The previous run's settings are recoverable.** CV22's provenance manifests record the
invocations used for its most recent targets: `beam_width 6`, `eval_budget 240`,
`cv_folds 0`, `max_rounds 10`, `min_features 4`, `final_top_k 4`,
`surrogate_model auto:xgb`, run one target at a time through `--dataset-prefix ... 
--limit-datasets 1`, with `--fit-timeout 1800` and `--no-improve-patience 999`. The last of
those disables the search's early stop, so the patience fix cannot fire under those settings
and its silence is not a defect. Matching them is what makes the new run comparable.

## Why the GP's epoch budget is unidentified: the feature set is small

Six explanations were refuted before this one, and this is the first that survives its own
test. The Gaussian process's epoch budget is unidentifiable **because the feature subsets the
search selects are small**, and the effect has a sharp threshold.

Measured on the CV22 tree with one factor changed -- the same `gp_01` config, the same
aggregation, the same hyperparameters, the target's own column list truncated to k columns
with the state feature always kept -- across all 14 targets, 70 fits:

    k    median plateau fraction   budget identified   plateau spans the whole range
    1           1.000                  0 of 14                 13 of 14
    2           1.000                  2 of 14                  8 of 14
    4           1.000                  2 of 14                 10 of 14
    7           0.020                 12 of 14                  0 of 14
    11          0.032                 10 of 14                  0 of 14

Spearman correlation between k and the plateau fraction is -0.791 over the 70 fits. Below
five columns the curve resolves nothing; at seven it resolves a minimum to within about 2%
of the scanned range on 12 of 14 targets.

That explains the 93% figure directly. The search is a backward elimination toward
`--min-features 4`, so the winners are small: their column counts are 1, 1, 1, 1, 1, 4, 5,
5, 5, 6, 7, 7, 9, 9, with 9 of 14 at six columns or fewer. The reported fits sit almost
entirely on the flat side of the threshold, which is why every survey of them found the
budget unidentified while the staged full-feature config resolves it on 11 of 14 targets.

**The caveat should therefore be stated narrowly.** It is not that the Gaussian process's
cross-validated budget does not work; it is that it cannot resolve a budget for a model with
four or fewer predictors, which is most of what the search selects. A plausible reading is
that such a model has too little to fit for the epoch count to matter much -- but that
reading is NOT fully supported: the leakage-free budget sweep found held-out R2 moving by up
to 1.29 across budgets on Turbidity, whose winner carries seven columns, so on at least some
targets the budget still matters where it is unidentified.

**One factor remains unisolated.** The 7- and 9-column winners are flat while the ladder's
k=7 is identified, and those winners are `gp_02`, `gp_03` and `gp_04` rather than `gp_01`, so
the kernel and aggregation variant modulates the threshold. Isolating that would need the
same ladder repeated per variant. It is not pursued here because it changes nothing about
the re-run: the Gaussian process keeps its random-fold cross-validation either way, and the
diagnostics now record identifiability per fit so the caveat can be quantified rather than
asserted.

## Four fixes applied after the pipeline walkthrough

**Replicate copies that carry no information are now discarded before training.** A window is
written out ten times so measurement uncertainty can be propagated, but the perturbation only
touches predictors carrying an uncertainty distribution, and the profiler-free set contains
none. The check that suppressed the copies at scoring time never reached the training folder,
so the transformer and the tree models trained on 520 rows spanning 52 windows while the
Gaussian process and the linear regressions trained on 52 rows spanning the same 52 windows.

`_drop_identical_replicates` in `e_Train` now collapses each group of copies to one
representative **when every member is numerically identical**, rewrites `train_files.txt` and
`test_files.txt` so the record matches what was loaded, and prints what it dropped. The test
is absence of variance rather than the predictor list, which keeps it safe on a
profiler-bearing run. Controlled by `data_split.drop_identical_replicates`, default true.

Verified on three arms: profiler-free transformer 520 -> 52 training rows and 220 -> 22 test
rows; profiler-free Gaussian process unchanged at 52, nothing dropped; profiler-bearing
transformer left at 130 rows over 13 windows with nothing dropped, because there the copies
genuinely differ.

This changes results. It is a tenfold reduction in the rows the transformer and the trees
train on, and the tree budget moves with it -- on Chromium the round estimator went from 44
rounds to 1 once the duplicates were gone, because the fold curves are computed on different
data. Numbers from earlier runs are not comparable across this change.

**The tree-count scan range no longer depends on how the run was invoked.** The rule scanned
300 rounds over five groups standalone and 19 rounds over three groups inside a sweep, and
two candidate explanations were tested and refuted without isolating the cause. The ceiling
now comes from the tuning search space's own upper bound for the number of trees
(`param_space.n_estimators.high`), which describes the search rather than a result and so
cannot have been overwritten by an earlier fit. Verified: the staged configuration and the
sweep-written configuration now both report a ceiling of 300 over five groups where they
previously disagreed. The gate inside `_train_xgb_model`, used when tuning is disabled, was
taking its fold count from a different setting and is now aligned with it.

**Every tree fit records how its budget was reached.** `budget_audit` in the training summary
carries the tuning cache actually consulted, whether it was a hit, and the scan ceiling, fold
count and grouping. The two cache locations are not a defect -- the sweep points
`cv_tuning.cache_path` at its own directory and a standalone run falls back to the dataset's
forecasts folder -- but a run that did not record which one it read could not be compared with
another, which is why the scan-range divergence took an investigation rather than one look at
an artifact.

**The transformer's fallback no longer consults the held-back split.** If the cross-validated
epoch estimate failed, the old fallback set the stopping source to `threshold`, which left
`max_epochs_override` unset -- and `train_transformer` then restores the epoch that scored
best on `testloader`, the held-back split. The fallback now pins the configured epoch ceiling
instead, so training runs its full length and the last epoch is kept. The path was never
taken in any run examined, which is precisely why it was worth closing.

## Correction: the GP never had duplicate rows to leak

An earlier entry in this file explained the Gaussian process's flat epoch curves as duplicate Monte
Carlo copies leaking across its cross-validation folds, and recorded the grouped-fold experiment that
refuted it. The refutation stands, but the premise was wrong, and the reason is worth stating because
it invalidates the reasoning rather than just the conclusion.

Each family's training folder is fixed in its configuration. `config_gp_*.yml` and the MLR path read
`samples/`, the unreplicated folder; `config_transformer_*.yml` and `config_xgb_*.yml` read
`mc_replicates/`. Verified on both CV22_profilerless and CV24_profilerless, and against the file lists
of completed runs: a stored Gaussian process fit trained on 52 rows spanning 52 windows, while a
transformer or tree fit on the same target trained on 520 rows spanning the same 52 windows.

So the Gaussian process trains one row per window and has nothing to group. Its ungrouped random folds
are equivalent to grouped ones, which is why adding a grouped scheme could not have improved anything
and why the measured result -- no change in identifiability -- was the only possible outcome. The
transformer and tree estimators, which do train on ten copies per window, already group by window.
Every family groups exactly when its data contains duplicates, and none of the three uses a
chronological split for choosing a training length.

What survives as a real finding is narrower and different: **the neural network and the tree family
train on ten identical copies of every window, for no added information.** The check that suppresses
the copies governs scoring only -- every model is scored on the 22 distinct held-back windows -- and
does not reach the choice of training folder. For the transformer this multiplies the gradient steps
per pass by ten; for the trees it inflates the sample count seen by the range-narrowing rule and by the
fold sizes, while that family's own settings tuning removes the copies, so the tuning and the fit
disagree about how much data exists.

## Duplicate rows leaking across GP folds: the original entry, premise now corrected

The Gaussian process's epoch budget is unidentified for 93% of fits, and five explanations
have already been refuted. A sixth was structural enough to look decisive, and it too
failed.

The mechanism is real. On a profiler-free predictor set the ten Monte Carlo replicates of a
segment are exact duplicates, by construction rather than by accident:
`UNCERTAINTY_DISTRIBUTION_FEATURES` in `h_RunMCFeatureSelectionSweep.py` contains only six
`Pfl -` channels, so a profiler-free candidate has nothing to perturb. Measured on the
staged tree, the non-profiler columns are identical across replicates on 236 of 236 segments
checked. The sweep trains on `mc_replicates`, so Arsenic's 520 training rows are 52 unique
segments repeated ten times.

And the GP was the only family whose folds ignored that. `_transformer_cv_estimate_epochs`
and every XGBoost fold builder call `_build_group_folds`, which collapses `_mc_\d+` through
`_base_sample_id`; the GP's default `random` scheme shuffled flattened row indices, so
roughly nine of every ten validation rows had an exact duplicate in the training fold. Fold
validation error cannot respond to overfitting under those conditions, which would explain a
flat curve without the epoch budget mattering at all.

A `grouped` scheme was added to `_gp_cv_estimate_epochs` and measured against `random` on
all 14 targets. It changed the chosen budget on 9 of them, so the folds genuinely differ and
the branch is doing what it claims. The plateau nonetheless narrowed on only 3 targets, was
identical on 11 and wider on none; the budget became identified on 1 target against 0; and
the plateau still spans the entire scanned range on 11 of 14. The held-out shortfall, a
diagnostic that must not be used to choose the scheme, also moved the wrong way -- median
+0.0101 against +0.0025 and mean +0.1056 against +0.0164, with Color going from 0.0000 to
+1.0543.

So the duplication is real, the GP's folds really did split it, and fixing that does not
make the budget identifiable. `random` stays the default; `grouped` remains available and
carries its refutation at the call site. The flatness is not yet explained.

Two consequences of the duplication stand regardless, and neither is addressed for this
run. Exact duplicate rows make the GP kernel matrix rank-deficient -- rank about 52 in a
520x520 matrix -- which is a plausible contributor to the jitter and convergence failures
already seen. And an exact GP is cubic in rows, so the tenfold duplication costs roughly a
thousandfold in the cubic term for no added information, making it the largest single cost
lever available. Changing it would alter `n_train` and break comparability, so it is
deliberately left alone and recorded as the next methodological step.

## Is the cross-validated budget worth its cost? Per family, measured

The rule was tested against the only fair alternative: a constant chosen LEAVE-ONE-TARGET-
OUT, best across the other thirteen and then applied to the held-out one. Picking the best
constant in hindsight from the same test scores would flatter it exactly as the GP's first
budget sweep flattered long budgets.

What the cross-validation buys is not average accuracy. It is protection against
overfitting, which is what this data punishes, and it shows in the worst case rather than in
any average:

    family / rule                worst R2   ruined(<-0.5)  below 0   best R2
    GP, CV                        -0.0129         0            1      0.5473
    GP, best constant (30)        -0.0280         0            2      0.4940
    GP, constant 300              -1.0431         2            4      0.5473
    XGB, CV round estimator       -0.0530         0            2      0.3687
    XGB, Optuna-tuned budget      -0.7075         1            5      0.3476
    XGB, constant 300             -0.7996         1            6      0.4016
    transformer, CV               -2.2165         2            8      0.4040
    transformer, constant 30      -2.1091         2            7      0.4058

Cost, measured back to back on the same configurations in one session, alternating the arms
so that machine load affects both equally. An earlier figure taken from two different runs
suggested the GP estimator was free; that was an artifact of comparing across runs and is
withdrawn:

    GP,          Arsenic  CV  5.2 s   pinned at 30  4.2 s    +1.0 s  (+24%)
    GP,          Nickel   CV  6.1 s   pinned at 30  4.1 s    +2.1 s  (+51%)
    transformer, Arsenic  CV 26.8 s   pinned at 30  8.8 s   +18.0 s (+204%)
    transformer, Nickel   CV 17.5 s   pinned at 30  8.5 s    +8.9 s (+105%)

The XGBoost estimator's cost is small in absolute terms: computing the full 5-fold,
300-round curve took a median of 3 s per target, against roughly 10 s for a train and
evaluate cycle.

**Gaussian process: keep the cross-validation.** It beats every constant on both the worst
case and the best case, is never more than 0.05 behind the best constant on any target, and
gains more than 0.05 on three (Zinc +0.54, Turbidity +0.18, Cadmium +0.11), for 24-51% more
time per fit.

**Transformer: replace it with a fixed budget.** A constant of 30 matches the CV rule on
every column above, and the per-target differences fall inside that target's own seed-noise
floor, while the estimator costs 105-204% more per fit -- the worst cost for the least
benefit of the three families. `fixed_epoch_budget` expresses this without falling into the
weight-restore path that would select on the test split. The constant should be justified
from the estimator's own clustering, which puts 11 of 14 targets between 22 and 37 epochs,
and never from the test scores; running the estimator on a calibration subset and then
fixing the value keeps that grounding at a fraction of the cost.

**XGBoost: keep the round estimator, and make it reachable again.** Its tiny budgets are
not the liability they first appear: they are the safest rule measured, with no ruined
targets and two negatives against five or six for either alternative, and a median shortfall
of +0.0096 against a constant 300's +0.0025. On weak signal, one to twenty rounds acts as
heavy regularisation -- pH scores +0.2049 at a single round and -0.4168 at three hundred.

This revises the earlier reading of those budgets as simply pathological. What stands is
narrower: the stored results are not reproducible from their configs, the feature search
ranked weakly-trained models across 3087 fits, and four targets would gain from more rounds
(Arsenic +0.17, Colony count +0.14, Copper +0.14).

**Fixed: the round estimator now runs alongside the tuning.** With `cv_tuning.enabled`
true the estimator had been unreachable, so a re-run would have fallen back to the tuned
`n_estimators` -- the weakest of the three rules on every risk column. The tuned path now
calls the estimator itself, on the tuned hyperparameters, and uses its answer as the budget.

It is called explicitly rather than by loosening the gate inside `_train_xgb_model`, because
that gate also requires a None `eval_set_override` and satisfying it would put the test
split back into `eval_set`. The explicit call keeps the train-only eval_set and
`disable_early_stopping=True`, so the leakage posture is unchanged: the estimator receives
`train_samples` and never sees the test split.

Two deliberate choices, both documented at the call site. The scan runs to the CONFIGURED
ceiling of 300 rounds rather than to the tuned `n_estimators`, because Optuna's value is a
hyperparameter tuned jointly with eight others and using it as the ceiling would cap the
budget at whatever it happened to pick -- 26 of a 300-round space on Arsenic; the measured
comparison that justifies keeping the rule was run over the full range. Folds and seed
follow the tuning's own settings, so the budget is estimated on the same partitions the
hyperparameters were chosen on.

Verified end to end on seven targets. The estimator fires, the tuned hyperparameters
survive, and `budget_mechanism` records `cv_round_estimator`:

    target            budget   new R2    stored R2   sweep best
    Cadmium              24    +0.3915    +0.3141     +0.3687
    Nickel               20    +0.3797    +0.4185     +0.3284
    Colony count          2    +0.2328    +0.3690     +0.2834
    Copper filtered      54    +0.1588    +0.2940     +0.4016
    pH                    1    +0.0983    +0.3198     +0.2049
    Arsenic              11    +0.0249    +0.1240     +0.1909
    Turbidity            21    -0.1460    +0.2362     +0.3305

Cadmium moves from -0.8054 before the fix to +0.3915, recovering the whole 1.2 R2 the broken
path was losing. The others land below their stored values, which is expected and is not a
regression: the stored numbers were single draws under a tuning cache that has since been
rewritten, and the differences here are seed and hyperparameter draws rather than a change
of rule. Exact reproduction of the stored numbers is not achievable and is not the goal.

**The budget mechanism is now recorded, not inferred.** `budget_mechanism` is written into
every XGBoost stop summary as one of `cv_round_estimator`, `configured_n_estimators`,
`validation_early_stopping` or `train_loss_plateau`. The stop reason CODE could never carry
this -- `cv_epoch_budget_exhausted` is reported both by the estimator and by any path that
merely disables early stopping, and that conflation is precisely why an entire arm's budget
mechanism went unnoticed. The transformer's stop reason was given the same treatment, so a
fixed budget no longer reports itself as CV-derived.

**The ninth invariant check was rewritten for the fixed state.** It had failed any run whose
stop text named the estimator while its config enabled tuning, which was correct for the
broken code and inverts the moment the bug is fixed -- that combination is now the intended
one. It compares the recorded `budget_mechanism` against the mechanism the config selects,
warns when a run asked for the estimator and silently fell back, and reports runs predating
the field as unverifiable rather than failed. Tested against a synthetic tree covering all
five cases, including a genuine mismatch, so it is known to discriminate rather than assumed
to; on the legacy CV22 tree it now reports 3381 runs as predating the field, with 0
failures.

## The summaries are stale for the two re-run targets

`data/output/CV22_profilerless/summaries/common_set_metrics.csv` was written 2026-09-01 at
06:55. Lead and Chromium were re-run that evening, between 17:56 and 20:29, after the
surrogate-degeneracy fix. Every one of their 91 final-stage fits postdates the summary, and
the summary's rows for them name run directories that no longer exist: for Lead the `gp`,
`transformer` and `xgb` runs are all absent, and for Chromium the `gp` run is. Only MLR
still resolves for Lead.

The archived numbers were real when written, but they are not reproducible from the current
tree and they do not reflect the re-run. `z8` has to be regenerated with an explicit
`--root` before Table 3 is rebuilt, and doing so will change those two rows.

## Outstanding

- **Decide what `--replicates` is for on the profiler-free arm.** Section 3.4 states
  that six replicates were run at each horizon, which is true but uninformative for 13 of
  the 14 targets. `--replicates 1` would give identical numbers and save ~14%.
- **Revise Table 3, Figure 12 and Section 3.3** for the five targets above, or re-run
  the sweep. This is the largest open item and it blocks submission.
- **The horizon sweep must be re-run after any selection change**, since it takes its
  winners from Table 3.
- **17 non-reproducing `xgb_01` candidates** and **8 unrecovered GP configurations** leave
  residual uncertainty on Colony count, Copper, E. coli, Total coliforms and pH.
- **`h_` has no timeout** around in-process training; a hung fit stalls a sweep silently.
- **`--seeds N` is untested end to end.**
- **Horizon skill columns are empty.** `lookahead_aggregate.csv` has `skill_v_*` as NaN
  because the horizon runs' `predictions.csv` carry no reference-forecast columns. Only
  the R^2 figures are used in the paper; `lookahead_skill_*` and
  `lookahead_time_to_zero_skill` from `z2` are not.
- **`pdflatex` is not installed here.** Table 3 now has eight `X` columns with `\hsize`
  coefficients summing to exactly 8.0, and the new Figure `fig:testpred` uses `\subfloat`
  from the MDPI class. Both need to be checked on the author's build.
- **The abstract and Introduction otherwise remain the author's.** The framing sentence
  at the end of the second Introduction paragraph and research question 2 were updated
  because the MLR reclassification made them factually wrong; nothing else was touched.

