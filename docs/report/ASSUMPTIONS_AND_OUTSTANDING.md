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

## Outstanding

- **Forecast horizon.** `k_RunHorizonSweep` has not been run on the profiler-free
  predictor set. Section 3.4, figure `fig:horizon`, the `m_{R^2}` definition in Section
  2.6, the horizon paragraph in Section 4, research question 4, the lead-time sentence in
  Section 2.3, and one clause of the abstract are all commented out in
  `manuscript.tex` pending that run. Each carries a LaTeX comment naming what to restore.
- **`pdflatex` is not installed here.** Table 3 now has eight `X` columns with `\hsize`
  coefficients summing to exactly 8.0, and the new Figure `fig:testpred` uses `\subfloat`
  from the MDPI class. Both need to be checked on the author's build.
- **The abstract and Introduction otherwise remain the author's.** The framing sentence
  at the end of the second Introduction paragraph and research question 2 were updated
  because the MLR reclassification made them factually wrong; nothing else was touched.

