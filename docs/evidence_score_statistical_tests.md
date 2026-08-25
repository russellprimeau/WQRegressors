# Statistical Tests for Evidence Score in summary_model_quality_matrix

This document summarizes the statistical tests used to evaluate the "Evidence Score" in the summary_model_quality_matrix figure, with mathematical notation in LaTeX.

## 1. Diebold-Mariano (DM) Test
Compares forecast accuracy between two models:
$$
DM = \frac{\bar{d}}{\sqrt{\frac{\hat{\gamma}_d(0) + 2 \sum_{k=1}^{h-1} \hat{\gamma}_d(k)}{n}}}
$$
Where:
- $\bar{d}$: mean difference in loss (e.g., RMSE) between models
- $\hat{\gamma}_d(k)$: autocovariance at lag $k$
- $n$: number of samples

## 2. Wilcoxon Signed-Rank Test
Non-parametric test for paired differences:
- Compute signed ranks of differences $d_i$ (excluding zeros).
- Test statistic $W$ is the sum of ranks for positive differences.

## 3. Sign Test
Counts wins/losses:
$$
n_{win} = \sum_i \mathbb{I}(d_i < 0) \qquad n_{loss} = \sum_i \mathbb{I}(d_i > 0)
$$
$p$-value from binomial test under $H_0: p=0.5$.

## 4. Cohen's d (Effect Size)
$$
d = \frac{\bar{d}}{s_d}
$$
Where $\bar{d}$ is the mean of differences, $s_d$ is their standard deviation.

## 5. Bootstrap Confidence Intervals
- Resample groups with replacement, recompute skill metric (e.g., $1 - \frac{\text{RMSE}_\text{model}}{\text{RMSE}_\text{baseline}}$).
- Compute percentiles for confidence intervals.

## 6. ANOVA Variance Components
Decompose variance into within-group and between-group components. Intraclass correlation coefficient (ICC):
$$
\text{ICC} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{between}} + \text{MS}_{\text{within}}}
$$

---

### Application
- The **Evidence Score** is a summary metric (e.g., mean skill improvement, probability model outperforms baseline) computed across independent test groups.
- Statistical tests (DM, Wilcoxon, Sign) are applied to the distribution of per-group differences in error or skill between model and baseline.
- **Bootstrap** is used to estimate confidence intervals and the probability that the model outperforms the baseline ($P(\text{skill} > 0)$).
- **Cohen's d** quantifies the effect size of the improvement.
- **ANOVA** may be used to estimate the fraction of variance attributable to group differences (relevant for uncertainty quantification).

---

For implementation details, see the relevant functions in `src/z1_FeaturePostProcess.py`.
