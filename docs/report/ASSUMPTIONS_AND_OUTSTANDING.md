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

