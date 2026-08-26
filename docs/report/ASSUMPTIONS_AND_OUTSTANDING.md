# Assumptions And Outstanding Content

## Assumptions applied in the manuscript draft

- The main paper should stay tightly focused on the predictive-methodology story and not attempt to document every script, sweep, or auxiliary diagnostic in the repository.
- The most defensible primary results are those summarized in `data/output/CV19/summaries_all/summary_best_model_performance.csv` and the companion summary figures already copied into the draft.
- The horizon figure currently used in the draft (`lookahead_time_to_baseline_r2.png`) appears to come from the older `CV14` horizon post-processing outputs rather than the `CV19` summary set. I kept it because it is the available report-ready figure already referenced by the manuscript, but it should be confirmed before submission.
- The author-contributions statement was converted from placeholders to a plausible draft based on repository ownership and funding metadata, but it still requires confirmation by the author team.
- The acknowledgments section was reduced to a neutral placeholder sentence that signals the need for a final institution-specific confirmation pass.

## Content still needing confirmation or addition

- Verify the exact lake-source wording in the study-area section:
  principal source vs. principal/secondary source.
- Confirm whether the manuscript should explicitly state the population served as "approximately 70,000" or a more current/official figure.
- Confirm the final wording for Norwegian regulatory context and whether a direct law or regulation citation should be added.
- Confirm the exact transformer architecture details if the paper should report layer count, head count, dimensionality, and loss definition explicitly in the main text rather than only conceptually.
- Confirm whether the GP implementation should cite `GPyTorch` explicitly with the intended bibliography entry.
- Confirm whether the manuscript should mention the exact number of predictor channels used in the final forecasting windows everywhere as `18`, since some parts of the repo describe 17 or 18 depending on representation.

## Figures that were refreshed from newer outputs

- `summary_model_quality_matrix.png`
- `ml_comparison_nrmse.png`
- `multi_target_importance_bars.png`
- `multi_target_feature_inclusion_heatmap.png`
- `lookahead_r2_comparison.png`
- `lookahead_nrmse_comparison.png`
- `lookahead_skill_comparison.png`
- `lookahead_r2_rate_bar.png`

`lookahead_time_to_baseline_r2.png` already matched the source file I checked, so it was not replaced.

## Important material intentionally left out of the main paper

- Most intermediate sweep artifacts, optimizer diagnostics, and per-target panels.
- Detailed implementation descriptions for every helper script under `src/utils/`.
- Particle-filter outputs and experimental branches not clearly tied to the core narrative of the current draft.
- Extensive uncertainty-analysis side outputs beyond what is already represented in the selected summary figures.

## Recommended next additions before submission

- Add one short paragraph in the discussion or limitations section that explicitly states the very small independent test-sample counts for several targets and the consequences for statistical power.
- Decide whether the paper should include one compact table of the 14 targets with best model, $R^2$, and skill vs. best baseline; right now those results are only visualized in figures.
- Confirm whether the horizon analysis should stay in the paper if it must rely on the older `CV14` figure lineage.
- Replace the acknowledgments placeholder with project-specific text.
- Review all citations once compiled, especially `Liu2024`, `Yan2024`, and any GP-related references.
