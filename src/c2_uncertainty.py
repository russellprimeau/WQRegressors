"""
Uncertainty Exploration Script (c2_uncertainty.py)

Comprehensive calibration uncertainty analysis using offset+gain linear models
and separate calibration point analysis.

Analyzes:
- Per-event offset+gain decomposition (gain, offset/drift, noise)
- Distribution fitting and normality testing
- Correlation with environmental predictors (Temperature, Timespan)
- Independence testing of residuals
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy import stats
from scipy.stats import norm, laplace, t, uniform, logistic, normaltest, shapiro, anderson, kstest
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
CORRECTIONS_DIR = Path(__file__).parent.parent / "data" / "output" / "calibration" / "summaries"
OUTPUT_DIR = CORRECTIONS_DIR
FIGURE_DPI = 300
PRACTICAL_R_THRESHOLD = 0.2
USE_ROBUST_PVALUE_FOR_COLOR = True  # Set to False to use OLS p-values for heatmap color scale

# Set style for plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.4



def clear_figure_titles(fig):
    """Remove all titles from a figure and its axes."""
    fig.suptitle('')
    for ax in fig.axes:
        ax.set_title('')


def normalize_sensor_name(csv_filename):
    """Normalize sensor names, excluding duplicates."""
    stem = Path(csv_filename).stem if isinstance(csv_filename, (str, Path)) else csv_filename
    stem = str(stem)
    
    if "Cond" in stem and "Sp Cond" not in stem:
        return None
    
    if "Sp Cond" in stem:
        return "Sp Cond (microS_cm)"
    
    return stem


def load_correction_data(csv_path):
    """Load calibration summary CSV."""
    return pd.read_csv(csv_path)


def apply_filtering_logic(raw_df):
    """
    Apply three filtering criteria to calibration data:
    1. Filter 1: Keep only Calibration Status == "Completed" (exclude "CompletedWithWarnings")
    2. Filter 2: Keep only rows where Last Calibration Time is not null AND
                 year(Last Calibration Time) >= year(Calibration Start Time)
    3. Filter 3: Nullify individual corrections where Stability Achieved != "Yes"
    """
    df = raw_df.copy()
    
    # Filter 1: Calibration Status == "Completed" (exact match)
    if 'Calibration Status' in df.columns:
        df = df[df['Calibration Status'] == 'Completed'].copy()
    
    # Filter 2: Last Calibration Time validation
    if 'Last Calibration Time' in df.columns and 'Calibration Start Time' in df.columns:
        # Parse both timestamp columns
        last_cal_time = pd.to_datetime(df['Last Calibration Time'], errors='coerce')
        cal_start_time = pd.to_datetime(df['Calibration Start Time'], errors='coerce')
        
        # Keep only rows where Last Calibration Time is not null
        mask_not_null = last_cal_time.notna()
        
        # Keep only rows where year(Last Calibration Time) >= year(Calibration Start Time)
        mask_year_valid = last_cal_time.dt.year >= cal_start_time.dt.year
        
        # Combine both conditions
        df = df[mask_not_null & mask_year_valid].copy()
    
    # Filter 3: Nullify corrections where Stability Achieved != "Yes"
    # Point 1
    if 'Stability Achieved' in df.columns and 'Correction1' in df.columns:
        df.loc[df['Stability Achieved'] != 'Yes', 'Correction1'] = np.nan
    
    # Point 2
    if 'Stability Achieved 2' in df.columns and 'Correction2' in df.columns:
        df.loc[df['Stability Achieved 2'] != 'Yes', 'Correction2'] = np.nan
    
    # Point 3
    if 'Stability Achieved 3' in df.columns and 'Correction3' in df.columns:
        df.loc[df['Stability Achieved 3'] != 'Yes', 'Correction3'] = np.nan
    
    return df.reset_index(drop=True)


def parse_timespan(timespan_str):
    """Convert timespan string to total seconds."""
    if pd.isna(timespan_str):
        return np.nan
    
    try:
        parts = str(timespan_str).split()
        days = 0
        seconds_part = None
        
        for i, part in enumerate(parts):
            if 'day' in part:
                days = int(parts[i-1]) if i > 0 else 0
            elif ':' in part:
                seconds_part = part
        
        if seconds_part:
            h, m, s = map(int, seconds_part.split(':'))
            return days * 86400 + h * 3600 + m * 60 + s
        return np.nan
    except:
        return np.nan


def extract_numeric_value(value_str):
    """Extract numeric value from strings with units."""
    if pd.isna(value_str):
        return np.nan
    
    try:
        cleaned = str(value_str)
        for unit in ['Â°C', '°C', '° C', '% Sat', '%Sat', 'QSU', 'RFU', 
                     'microS_cm', 'microS/cm', 'mmHg', 'mg/L', 'FNU', 'NTU']:
            cleaned = cleaned.replace(unit, '')
        return float(cleaned.strip())
    except:
        return np.nan


def fit_offset_gain_model(row, sensor_name):
    """
    Fit offset+gain model for a single calibration event.
    Model: Error_i = Offset + Gain * PostCal_i + epsilon_i
    Where Error_i = PostCal_i - PreCal_i (correction needed)
         PostCal_i = Post Calibration Value (measured value, independent variable)
    Returns dict with offset, gain, noise_variance, fit_type, residuals.
    """
    # Extract calibration point pairs (Pre/PostCal values)
    point_data = []
    
    # Identify all calibration points in this event
    for i in range(1, 4):
        if i == 1:
            post_col = 'Post Calibration Value'
            pre_col = 'Pre Calibration Value'
        else:
            post_col = f'Post Calibration Value {i}'
            pre_col = f'Pre Calibration Value {i}'
        
        if post_col in row.index and pre_col in row.index:
            post_val = extract_numeric_value(row[post_col])
            pre_val = extract_numeric_value(row[pre_col])
            
            if not np.isnan(post_val) and not np.isnan(pre_val):
                error = post_val - pre_val  # Correction needed
                point_data.append((post_val, error))
    
    if len(point_data) < 1:
        return None
    
    post_cals = np.array([p[0] for p in point_data])
    errors = np.array([p[1] for p in point_data])
    
    # Single point: offset only, no gain
    if len(post_cals) == 1:
        noise_var = errors[0] ** 2
        
        return {
            'Offset': errors[0],
            'Gain': 0.0,
            'Noise_Variance': noise_var,
            'Noise_Std': np.sqrt(noise_var),
            'Model_F_stat': np.nan,
            'Model_p_value': np.nan,
            'Model_Significant': False,
            'N_Parameters': 1,
            'Fit_Type': 'Single_Point',
            'N_Points': 1,
            'Residuals': errors
        }
    
    # Multi-point: fit line Error = Offset + Gain * PostCal
    try:
        coeffs = np.polyfit(post_cals, errors, 1)
        gain = coeffs[0]
        offset = coeffs[1]
        
        # Calculate residuals and noise variance
        y_fit = gain * post_cals + offset
        residuals = errors - y_fit
        
        # Unbiased noise variance estimate
        n = len(post_cals)
        dof = n - 2  # Two parameters: offset and gain
        noise_var = np.sum(residuals ** 2) / dof if dof > 0 else np.nan
        
        # Calculate R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((errors - np.mean(errors)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # F-test: offset+gain model (2 parameters) vs null model (0 parameters)
        # H0: Error = constant (mean), H1: Error = Offset + Gain*PostCal
        ss_null = ss_tot  # Total sum of squares (null model)
        ss_fit = ss_res   # Residual sum of squares (fitted model)
        
        # F-statistic with df1=2 (2 parameters), df2=n-2 (residual DOF)
        f_stat = ((ss_null - ss_fit) / 2) / (ss_fit / dof) if dof > 0 and ss_fit > 0 else np.nan
        
        # P-value: probability of observing this F-stat or larger under null hypothesis
        if not np.isnan(f_stat) and f_stat >= 0:
            p_value = stats.f.sf(f_stat, 2, dof)  # Survival function = 1 - CDF
        else:
            p_value = np.nan
        
        # Model is significant if p-value < 0.05
        model_significant = not np.isnan(p_value) and p_value < 0.05
        
        return {
            'Offset': offset,
            'Gain': gain,
            'Noise_Variance': noise_var,
            'Noise_Std': np.sqrt(noise_var),
            'R_squared': r_squared,
            'Model_F_stat': f_stat,
            'Model_p_value': p_value,
            'Model_Significant': model_significant,
            'N_Parameters': 2,
            'Fit_Type': 'Multi_Point',
            'N_Points': len(post_cals),
            'Residuals': residuals
        }
    except:
        return None


def calculate_offset_gain_results(raw_df, sensor_name):
    """
    Fit offset+gain model for each calibration event.
    Returns DataFrame with results.
    """
    results = []
    
    for idx, row in raw_df.iterrows():
        model = fit_offset_gain_model(row, sensor_name)
        if model is None:
            continue
        
        model['Event_Index'] = idx
        
        # Extract predictors
        if 'Temperature' in raw_df.columns:
            temp = extract_numeric_value(row['Temperature'])
            model['Temperature'] = temp if not np.isnan(temp) else np.nan
        
        if 'Timespan' in raw_df.columns:
            ts = parse_timespan(row['Timespan'])
            model['Timespan_seconds'] = ts if not np.isnan(ts) else np.nan
        
        results.append(model)
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def test_predictor_significance(y_data, x_data, y_name, x_name):
    """
    Test whether a predictor variable significantly affects a response variable.
    Performs linear regression: y = b0 + b1*x + error
    Returns dict with regression coefficients, t-stat, p-value, R².
    """
    result = {}
    
    if len(y_data) < 3:
        return result
    
    try:
        n = len(y_data)
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        
        # Store results
        result[f'{y_name}_vs_{x_name}_slope'] = slope
        result[f'{y_name}_vs_{x_name}_intercept'] = intercept
        result[f'{y_name}_vs_{x_name}_stderr'] = std_err
        result[f'{y_name}_vs_{x_name}_r'] = r_value
        result[f'{y_name}_vs_{x_name}_r2'] = r_value ** 2
        result[f'{y_name}_vs_{x_name}_pvalue'] = p_value
        result[f'{y_name}_vs_{x_name}_significant'] = p_value < 0.05
        result[f'{y_name}_vs_{x_name}_n'] = n
        result[f'{y_name}_vs_{x_name}_practical'] = abs(r_value) >= PRACTICAL_R_THRESHOLD
        
        # Compute t-statistic manually
        if std_err > 0:
            t_stat = slope / std_err
            result[f'{y_name}_vs_{x_name}_tstat'] = t_stat

        # Spearman correlation (robust to non-linearity/outliers)
        rho, rho_p = stats.spearmanr(x_data, y_data)
        result[f'{y_name}_vs_{x_name}_spearman_r'] = rho
        result[f'{y_name}_vs_{x_name}_spearman_p'] = rho_p

        # Robust regression (Theil-Sen) with robust significance proxy
        try:
            ts_slope, ts_intercept, _, _ = stats.theilslopes(y_data, x_data)
            result[f'{y_name}_vs_{x_name}_robust_slope'] = ts_slope
            result[f'{y_name}_vs_{x_name}_robust_intercept'] = ts_intercept
            result[f'{y_name}_vs_{x_name}_robust_method'] = 'theil-sen'
        except:
            pass

        result[f'{y_name}_vs_{x_name}_robust_pvalue'] = rho_p
        result[f'{y_name}_vs_{x_name}_robust_r'] = rho
        result[f'{y_name}_vs_{x_name}_robust_significant'] = rho_p < 0.05
        result[f'{y_name}_vs_{x_name}_robust_practical'] = abs(rho) >= PRACTICAL_R_THRESHOLD
    except:
        pass
    
    return result


def test_offset_gain_statistics(og_df, sensor_name):
    """
    Test four key hypotheses on offset, gain, and noise.
    H1_Offset: Do offset estimates vary significantly across events?
    H1_Gain: Do gain estimates vary significantly across events?
    H1_Noise: Is the noise distribution stable across events?
    H2/H3: Do drift (offset/gain) or noise correlate with predictors?
    Tests predictor significance using linear regression (not just correlation).
    """
    stats_dict = {'Sensor': sensor_name}
    
    if len(og_df) < 2:
        return stats_dict

    multi_df = og_df[og_df['Fit_Type'] == 'Multi_Point']
    if len(multi_df) > 0:
        sig_count = int(multi_df['Model_Significant'].sum())
        sig_total = int(len(multi_df))
        stats_dict['Model_Significance_Rate'] = sig_count / sig_total
        stats_dict['Model_Significance_Count'] = sig_count
        stats_dict['Model_Significance_Total'] = sig_total
    
    # H1_Offset: Offset Constancy
    offsets = og_df['Offset'].dropna()
    if len(offsets) > 1:
        offset_mean = offsets.mean()
        offset_std = offsets.std()
        offset_cv = offset_std / np.abs(offset_mean) if offset_mean != 0 else np.nan
        stats_dict['Offset_Mean'] = offset_mean
        stats_dict['Offset_Std'] = offset_std
        stats_dict['Offset_CV'] = offset_cv
        stats_dict['Offset_N'] = len(offsets)
    
    # H1_Gain: Gain Constancy (multi-point only)
    gains = og_df[og_df['Fit_Type'] == 'Multi_Point']['Gain'].dropna()
    if len(gains) > 1:
        gain_mean = gains.mean()
        gain_std = gains.std()
        gain_cv = gain_std / np.abs(gain_mean) if gain_mean != 0 else np.nan
        stats_dict['Gain_Mean'] = gain_mean
        stats_dict['Gain_Std'] = gain_std
        stats_dict['Gain_CV'] = gain_cv
        stats_dict['Gain_N'] = len(gains)
    
    # H1_Noise: Noise Stability across events
    noise_vars = og_df['Noise_Variance'].dropna()
    if len(noise_vars) > 1:
        noise_mean = noise_vars.mean()
        noise_std = noise_vars.std()
        noise_cv = noise_std / noise_mean if noise_mean > 0 else np.nan
        stats_dict['Noise_Variance_Mean'] = noise_mean
        stats_dict['Noise_Variance_Std'] = noise_std
        stats_dict['Noise_Variance_CV'] = noise_cv
        stats_dict['Noise_N'] = len(noise_vars)
        
        # Test: Levene's test for variance homogeneity
        # Prepare residuals from each event
        residual_groups = []
        for residuals in og_df['Residuals'].dropna():
            if isinstance(residuals, np.ndarray) and len(residuals) > 0:
                residual_groups.append(residuals)
        
        if len(residual_groups) > 1:
            try:
                levene_stat, levene_p = stats.levene(*residual_groups)
                stats_dict['Noise_Levene_stat'] = levene_stat
                stats_dict['Noise_Levene_p'] = levene_p
            except:
                pass
        
        # Shapiro-Wilk test on combined residuals
        all_residuals = []
        for residuals in og_df['Residuals'].dropna():
            if isinstance(residuals, np.ndarray):
                all_residuals.extend(residuals)
        
        if len(all_residuals) > 3:
            try:
                sw_stat, sw_p = shapiro(all_residuals)
                stats_dict['Noise_Shapiro_Wilk_stat'] = sw_stat
                stats_dict['Noise_Shapiro_Wilk_p'] = sw_p
            except:
                pass
    
    # H2/H3: Predictor Significance Tests (Linear Regression)
    # Use all events for Offset/Noise (more data), multi-point only for Gain (required)
    
    # H2a: Offset vs Temperature
    og_temp = og_df[['Offset', 'Temperature']].dropna()
    if len(og_temp) > 2:
        reg_results = test_predictor_significance(og_temp['Offset'].values, og_temp['Temperature'].values, 
                                                  'Offset', 'Temperature')
        stats_dict.update(reg_results)
    
    # H2a: Gain vs Temperature (multi-point only)
    og_gain_temp = og_df[og_df['Fit_Type'] == 'Multi_Point'][['Gain', 'Temperature']].dropna()
    if len(og_gain_temp) > 2:
        reg_results = test_predictor_significance(og_gain_temp['Gain'].values, og_gain_temp['Temperature'].values,
                                                  'Gain', 'Temperature')
        stats_dict.update(reg_results)
    
    # H2a: Noise vs Temperature
    og_noise_temp = og_df[['Noise_Variance', 'Temperature']].dropna()
    if len(og_noise_temp) > 2:
        reg_results = test_predictor_significance(og_noise_temp['Noise_Variance'].values, og_noise_temp['Temperature'].values,
                                                  'Noise', 'Temperature')
        stats_dict.update(reg_results)
    
    # H3b: Offset vs Timespan
    og_time = og_df[['Offset', 'Timespan_seconds']].dropna()
    if len(og_time) > 2:
        reg_results = test_predictor_significance(og_time['Offset'].values, og_time['Timespan_seconds'].values,
                                                  'Offset', 'Timespan')
        stats_dict.update(reg_results)
    
    # H3b: Gain vs Timespan (multi-point only)
    og_gain_time = og_df[og_df['Fit_Type'] == 'Multi_Point'][['Gain', 'Timespan_seconds']].dropna()
    if len(og_gain_time) > 2:
        reg_results = test_predictor_significance(og_gain_time['Gain'].values, og_gain_time['Timespan_seconds'].values,
                                                  'Gain', 'Timespan')
        stats_dict.update(reg_results)
    
    # H3b: Noise vs Timespan
    og_noise_time = og_df[['Noise_Variance', 'Timespan_seconds']].dropna()
    if len(og_noise_time) > 2:
        reg_results = test_predictor_significance(og_noise_time['Noise_Variance'].values, og_noise_time['Timespan_seconds'].values,
                                                  'Noise', 'Timespan')
        stats_dict.update(reg_results)
    
    return stats_dict


def prepare_separate_analysis(raw_df, sensor_name):
    """
    Prepare data for separate analysis of each calibration point.
    Returns dict of {point_name: dataframe}.
    """
    separate_data = {}
    
    for point_num, corr_col in enumerate([('Correction1', 'Standard', 'Post Calibration Value'),
                                           ('Correction2', 'Standard 2', 'Post Calibration Value 2'),
                                           ('Correction3', 'Standard 3', 'Post Calibration Value 3')], 1):
        corr_col_name = corr_col[0]
        std_col_name = corr_col[1]
        cal_col_name = corr_col[2]
        
        if corr_col_name not in raw_df.columns:
            continue
        
        corr_series = raw_df[corr_col_name]
        if isinstance(corr_series, pd.DataFrame):
            corr_series = corr_series.iloc[:, 0]

        df = raw_df[corr_series.notna()].copy()
        if len(df) < 1:
            continue
        
        # Extract numeric correction values
        df['Error'] = pd.to_numeric(corr_series.loc[df.index], errors='coerce')
        df = df.dropna(subset=['Error'])
        
        if len(df) < 1:
            continue
        
        # Extract temperature and timespan
        if 'Temperature' in df.columns:
            df['Temperature_numeric'] = df['Temperature'].apply(extract_numeric_value)
        else:
            df['Temperature_numeric'] = np.nan
        
        if 'Timespan' in df.columns:
            df['Timespan_seconds'] = df['Timespan'].apply(parse_timespan)
        else:
            df['Timespan_seconds'] = np.nan
        
        point_name = f'Calibration Point {point_num}'
        separate_data[point_name] = df
    
    return separate_data


def test_normality(data):
    """Perform normality tests."""
    results = {}
    
    if len(data) < 3:
        return results
    
    # Shapiro-Wilk
    if len(data) <= 5000:
        stat_sw, p_sw = shapiro(data)
        results['Shapiro_Wilk_stat'] = stat_sw
        results['Shapiro_Wilk_p'] = p_sw
    
    # Anderson-Darling
    result_ad = anderson(data)
    results['Anderson_Darling_stat'] = result_ad.statistic
    results['Anderson_Darling_critical_5pct'] = result_ad.critical_values[2] if len(result_ad.critical_values) > 2 else np.nan
    
    # Kolmogorov-Smirnov
    data_std = (data - np.mean(data)) / np.std(data) if np.std(data) > 0 else data
    stat_ks, p_ks = kstest(data_std, 'norm')
    results['KS_stat'] = stat_ks
    results['KS_p'] = p_ks
    
    # D'Agostino-Pearson
    stat_dp, p_dp = normaltest(data)
    results['DAgostino_stat'] = stat_dp
    results['DAgostino_p'] = p_dp
    
    return results


def fit_distributions(data):
    """Fit multiple distributions and compare AIC."""
    if len(data) < 3:
        return {}
    
    results = {}
    
    try:
        # Normal
        params_norm = norm.fit(data)
        loglik_norm = np.sum(norm.logpdf(data, *params_norm))
        aic_norm = -2 * loglik_norm + 2 * len(params_norm)
        results['Normal_AIC'] = aic_norm
    except:
        results['Normal_AIC'] = np.nan
    
    try:
        # Laplace
        params_laplace = laplace.fit(data)
        loglik_laplace = np.sum(laplace.logpdf(data, *params_laplace))
        aic_laplace = -2 * loglik_laplace + 2 * len(params_laplace)
        results['Laplace_AIC'] = aic_laplace
    except:
        results['Laplace_AIC'] = np.nan
    
    try:
        # Student t
        params_t = t.fit(data)
        loglik_t = np.sum(t.logpdf(data, *params_t))
        aic_t = -2 * loglik_t + 2 * len(params_t)
        results['StudentT_AIC'] = aic_t
    except:
        results['StudentT_AIC'] = np.nan
    
    try:
        # Uniform
        params_uniform = uniform.fit(data)
        loglik_uniform = np.sum(uniform.logpdf(data, *params_uniform))
        aic_uniform = -2 * loglik_uniform + 2 * len(params_uniform)
        results['Uniform_AIC'] = aic_uniform
    except:
        results['Uniform_AIC'] = np.nan
    
    try:
        # Logistic
        params_logistic = logistic.fit(data)
        loglik_logistic = np.sum(logistic.logpdf(data, *params_logistic))
        aic_logistic = -2 * loglik_logistic + 2 * len(params_logistic)
        results['Logistic_AIC'] = aic_logistic
    except:
        results['Logistic_AIC'] = np.nan
    
    return results


def test_independence(data):
    """Test independence of residuals."""
    results = {}
    
    if len(data) < 3:
        return results
    
    # Durbin-Watson test (simplified)
    if len(data) > 1:
        diffs = np.diff(data)
        ss_diffs = np.sum(diffs ** 2)
        ss_data = np.sum((data - np.mean(data)) ** 2)
        dw = ss_diffs / ss_data if ss_data > 0 else np.nan
        results['Durbin_Watson'] = dw
    
    return results


def calculate_distribution_stats(data):
    """Calculate distribution statistics."""
    return {
        'Mean': np.mean(data),
        'Std': np.std(data, ddof=1),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'N': len(data)
    }


def analyze_separate_point(df, sensor_name, point_name):
    """Analyze a single calibration point's data."""
    data = df['Error'].values
    point_stats = {}
    
    # Distribution stats
    dist_stats = calculate_distribution_stats(data)
    point_stats.update(dist_stats)
    
    # Normality tests
    norm_tests = test_normality(data)
    point_stats.update(norm_tests)
    
    # Distribution fitting
    dist_fits = fit_distributions(data)
    point_stats.update(dist_fits)
    
    # Temperature correlation
    if 'Temperature_numeric' in df.columns:
        temp_data = df[['Error', 'Temperature_numeric']].dropna()
        if len(temp_data) > 2:
            r_temp, p_temp = stats.pearsonr(temp_data['Error'], temp_data['Temperature_numeric'])
            point_stats['Temp_Correlation_r'] = r_temp
            point_stats['Temp_Correlation_p'] = p_temp
            point_stats['Temp_Correlation_R2'] = r_temp ** 2
    
    # Timespan correlation
    if 'Timespan_seconds' in df.columns:
        time_data = df[['Error', 'Timespan_seconds']].dropna()
        if len(time_data) > 2:
            r_time, p_time = stats.pearsonr(time_data['Error'], time_data['Timespan_seconds'])
            point_stats['Time_Correlation_r'] = r_time
            point_stats['Time_Correlation_p'] = p_time
            point_stats['Time_Correlation_R2'] = r_time ** 2
    
    # Independence tests
    indep_tests = test_independence(data)
    point_stats.update(indep_tests)
    
    return point_stats


def test_distribution_fit(data, data_name):
    """
    Test goodness-of-fit for both normal and Student's t distributions.
    Returns dictionary with test statistics, AIC comparison, and fitted parameters.
    
    Designed for small samples (n=20-30) typical of calibration event populations.
    """
    results = {
        'component': data_name,
        'n': len(data)
    }
    
    if len(data) < 3:
        return results
    
    # Remove NaN/inf
    clean_data = data[np.isfinite(data)]
    if len(clean_data) < 3:
        return results
    
    results['n_clean'] = len(clean_data)
    
    # === FIT NORMAL DISTRIBUTION ===
    try:
        norm_params = stats.norm.fit(clean_data)
        results['norm_loc'] = norm_params[0]
        results['norm_scale'] = norm_params[1]
        
        # Log-likelihood and AIC
        norm_loglik = np.sum(stats.norm.logpdf(clean_data, *norm_params))
        results['norm_loglik'] = norm_loglik
        results['norm_aic'] = 2 * 2 - 2 * norm_loglik  # k=2 parameters
    except:
        pass
    
    # === FIT STUDENT'S T DISTRIBUTION ===
    try:
        t_params = stats.t.fit(clean_data)
        results['t_df'] = t_params[0]
        results['t_loc'] = t_params[1]
        results['t_scale'] = t_params[2]
        
        # Log-likelihood and AIC
        t_loglik = np.sum(stats.t.logpdf(clean_data, *t_params))
        results['t_loglik'] = t_loglik
        results['t_aic'] = 2 * 3 - 2 * t_loglik  # k=3 parameters
    except:
        pass
    
    # === AIC COMPARISON ===
    if 'norm_aic' in results and 't_aic' in results:
        results['aic_diff'] = results['norm_aic'] - results['t_aic']  # Positive = t is better
        results['preferred'] = 't' if results['aic_diff'] > 2 else ('norm' if results['aic_diff'] < -2 else 'equivalent')
    
    # === SHAPIRO-WILK TEST (for normality) ===
    if len(clean_data) >= 3:
        try:
            sw_stat, sw_p = stats.shapiro(clean_data)
            results['shapiro_stat'] = sw_stat
            results['shapiro_p'] = sw_p
            results['shapiro_reject_normality'] = sw_p < 0.05
        except:
            pass
    
    # === ANDERSON-DARLING TEST (for normality) ===
    try:
        ad_result = stats.anderson(clean_data, dist='norm')
        results['anderson_stat'] = ad_result.statistic
        # Critical values at [15%, 10%, 5%, 2.5%, 1%]
        results['anderson_critical_5pct'] = ad_result.critical_values[2]
        results['anderson_reject_normality'] = ad_result.statistic > ad_result.critical_values[2]
    except:
        pass
    
    # === Q-Q CORRELATION COEFFICIENTS ===
    # For normal
    if len(clean_data) > 2:
        try:
            sorted_data = np.sort(clean_data)
            n = len(sorted_data)
            theoretical_norm = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            results['qq_norm_r'] = np.corrcoef(theoretical_norm, sorted_data)[0, 1]
        except:
            pass
    
    # For fitted t-distribution
    if 't_df' in results and len(clean_data) > 2:
        try:
            sorted_data = np.sort(clean_data)
            n = len(sorted_data)
            theoretical_t = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=results['t_df'])
            results['qq_t_r'] = np.corrcoef(theoretical_t, sorted_data)[0, 1]
        except:
            pass
    
    # === KURTOSIS AND SKEWNESS ===
    results['kurtosis'] = stats.kurtosis(clean_data)  # Excess kurtosis (0 for normal)
    results['skewness'] = stats.skew(clean_data)
    
    return results


def create_error_distribution_visualizations(og_df, sensor_name, output_dir=OUTPUT_DIR):
    """
    Create comprehensive visualizations of error source distributions.
    Shows offset, gain, and noise histograms and Q-Q plots comparing to normal and Student's t distributions.
    """
    if len(og_df) < 2:
        return
    
    has_independent_noise = 'N_Points' in og_df.columns and (og_df['N_Points'] >= 3).any()
    n_rows = 4 if has_independent_noise else 3
    fig_height = 15 if has_independent_noise else 12
    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.32, wspace=0.28)
    
    fig.suptitle(f'{sensor_name} - Error Decomposition', fontsize=14, fontweight='bold', y=0.995)
    
    # ===== ROW 0: TOTAL ERROR (not decomposed) =====
    
    # Extract raw errors from residuals
    all_errors = []
    all_point_labels = []
    for residuals in og_df['Residuals'].dropna():
        if isinstance(residuals, np.ndarray):
            for i, err in enumerate(residuals):
                all_errors.append(err)
                all_point_labels.append(f'Point {i+1}')
    
    if len(all_errors) > 0:
        all_errors = np.array(all_errors)
        all_point_labels = np.array(all_point_labels)
        
        # Total Error Histogram (color-coded by point)
        ax = fig.add_subplot(gs[0, 0])
        colors_map = {'Point 1': '#1f77b4', 'Point 2': '#ff7f0e', 'Point 3': '#2ca02c'}
        
        # Calculate shared bin edges for all points
        n_bins = max(5, len(all_errors)//10)
        bin_edges = np.histogram_bin_edges(all_errors, bins=n_bins)
        
        for point_label in ['Point 1', 'Point 2', 'Point 3']:
            point_errors = all_errors[all_point_labels == point_label]
            if len(point_errors) > 0:
                ax.hist(point_errors, bins=bin_edges, alpha=0.6, 
                       color=colors_map[point_label], edgecolor='black', linewidth=0.5, 
                       label=f'{point_label}, n = {len(point_errors)}')
        ax.set_xlabel('Total Error')
        ax.set_ylabel('Count')
        ax.set_title('Total Error Distribution', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
        
        # Total Error Q-Q Plot vs Normal (separate for each point)
        ax = fig.add_subplot(gs[0, 1])
        colors_map = {'Point 1': '#1f77b4', 'Point 2': '#ff7f0e', 'Point 3': '#2ca02c'}
        
        for point_label in ['Point 1', 'Point 2', 'Point 3']:
            point_errors = all_errors[all_point_labels == point_label]
            if len(point_errors) > 2:
                # Calculate theoretical quantiles manually
                sorted_errors = np.sort(point_errors)
                n = len(sorted_errors)
                theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
                ax.scatter(theoretical_quantiles, sorted_errors, alpha=0.6, s=30, 
                          color=colors_map[point_label], label=f'{point_label}, n = {n}', edgecolors='none')
        
        # Add reference line
        if len(all_errors) > 2:
            all_sorted = np.sort(all_errors)
            all_theoretical = stats.norm.ppf((np.arange(1, len(all_sorted) + 1) - 0.5) / len(all_sorted))
            # Fit line through all points for reference
            slope, intercept = np.polyfit(all_theoretical, all_sorted, 1)
            x_ref = np.array([all_theoretical.min(), all_theoretical.max()])
            ax.plot(x_ref, slope * x_ref + intercept, 'r-', linewidth=1.5, alpha=0.7, label='Reference')
        
        ax.set_xlabel('Normal Quantiles', fontweight='bold')
        ax.set_ylabel('Sample Quantiles', fontweight='bold')
        ax.set_title('Q-Q Plot vs Normal', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        
        # Total Error Q-Q Plot vs Student's t
        ax = fig.add_subplot(gs[0, 2])
        # Fit t-distribution to all errors
        if len(all_errors) > 3:
            try:
                t_params = stats.t.fit(all_errors)
                t_df, t_loc, t_scale = t_params
                
                for point_label in ['Point 1', 'Point 2', 'Point 3']:
                    point_errors = all_errors[all_point_labels == point_label]
                    if len(point_errors) > 2:
                        sorted_errors = np.sort(point_errors)
                        n = len(sorted_errors)
                        theoretical_quantiles = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=t_df)
                        ax.scatter(theoretical_quantiles, sorted_errors, alpha=0.6, s=30, 
                                  color=colors_map[point_label], label=f'{point_label}, n = {n}', edgecolors='none')
                
                # Add reference line
                all_sorted = np.sort(all_errors)
                all_theoretical_t = stats.t.ppf((np.arange(1, len(all_sorted) + 1) - 0.5) / len(all_sorted), df=t_df)
                slope, intercept = np.polyfit(all_theoretical_t, all_sorted, 1)
                x_ref = np.array([all_theoretical_t.min(), all_theoretical_t.max()])
                ax.plot(x_ref, slope * x_ref + intercept, 'r-', linewidth=1.5, alpha=0.7, label='Reference')
                
                ax.set_xlabel(f"Student's t Quantiles (df={t_df:.1f})", fontweight='bold')
                ax.set_ylabel('Sample Quantiles', fontweight='bold')
                ax.set_title("Q-Q Plot vs Student's t", fontweight='bold', fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.25)
                ax.set_axisbelow(True)
            except:
                ax.text(0.5, 0.5, 'Unable to fit t-distribution', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # ===== ROW 1: OFFSET Distribution =====
    
    # Offset Histogram
    ax = fig.add_subplot(gs[1, 0])
    offsets = og_df['Offset'].dropna()
    if len(offsets) > 1:
        ax.hist(offsets, bins=max(5, len(offsets)//3), alpha=0.75, color='#1f77b4', edgecolor='black', linewidth=0.7)
        ax.axvline(offsets.mean(), color='#d62728', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.axvline(np.median(offsets), color='#ff7f0e', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.set_xlabel('Offset')
        ax.set_ylabel('Count')
        ax.set_title('Offset Distribution', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    
    # Offset Q-Q Plot vs Normal
    ax = fig.add_subplot(gs[1, 1])
    if len(offsets) > 2:
        stats.probplot(offsets, dist="norm", plot=ax)
        ax.set_title('Offset Q-Q vs Normal', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normal Quantiles', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    
    # Offset Q-Q Plot vs Student's t
    ax = fig.add_subplot(gs[1, 2])
    if len(offsets) > 3:
        try:
            t_params = stats.t.fit(offsets)
            t_df = t_params[0]
            sorted_data = np.sort(offsets)
            n = len(sorted_data)
            theoretical_t = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=t_df)
            ax.scatter(theoretical_t, sorted_data, alpha=0.6, s=40, color='#1f77b4')
            
            # Reference line
            slope, intercept = np.polyfit(theoretical_t, sorted_data, 1)
            x_ref = np.array([theoretical_t.min(), theoretical_t.max()])
            ax.plot(x_ref, slope * x_ref + intercept, 'r-', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel(f"Student's t Quantiles (df={t_df:.1f})", fontweight='bold')
            ax.set_ylabel('Sample Quantiles', fontweight='bold')
            ax.set_title("Offset Q-Q vs Student's t", fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
        except:
            ax.axis('off')
    else:
        ax.axis('off')
    
    # ===== ROW 2: GAIN Distribution =====
    
    # Gain Histogram (multi-point only)
    ax = fig.add_subplot(gs[2, 0])
    gains = og_df[og_df['Fit_Type'] == 'Multi_Point']['Gain'].dropna()
    if len(gains) > 1:
        ax.hist(gains, bins=max(5, len(gains)//3), alpha=0.75, color='#ff7f0e', edgecolor='black', linewidth=0.7)
        ax.axvline(gains.mean(), color='#d62728', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.axvline(np.median(gains), color='#2ca02c', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.set_xlabel('Gain')
        ax.set_ylabel('Count')
        ax.set_title('Gain Distribution', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, 'Insufficient Multi-Point Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Gain Q-Q Plot vs Normal
    ax = fig.add_subplot(gs[2, 1])
    if len(gains) > 2:
        stats.probplot(gains, dist="norm", plot=ax)
        ax.set_title('Gain Q-Q vs Normal', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normal Quantiles', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')
    
    # Gain Q-Q Plot vs Student's t
    ax = fig.add_subplot(gs[2, 2])
    if len(gains) > 3:
        try:
            t_params = stats.t.fit(gains)
            t_df = t_params[0]
            sorted_data = np.sort(gains)
            n = len(sorted_data)
            theoretical_t = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=t_df)
            ax.scatter(theoretical_t, sorted_data, alpha=0.6, s=40, color='#ff7f0e')
            
            # Reference line
            slope, intercept = np.polyfit(theoretical_t, sorted_data, 1)
            x_ref = np.array([theoretical_t.min(), theoretical_t.max()])
            ax.plot(x_ref, slope * x_ref + intercept, 'r-', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel(f"Student's t Quantiles (df={t_df:.1f})", fontweight='bold')
            ax.set_ylabel('Sample Quantiles', fontweight='bold')
            ax.set_title("Gain Q-Q vs Student's t", fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
        except:
            ax.axis('off')
    else:
        ax.axis('off')
    
    if has_independent_noise:
        # ===== ROW 3: NOISE Distribution =====
        
        # Noise Variance Histogram
        ax = fig.add_subplot(gs[3, 0])
        noise_vars = og_df['Noise_Variance'].dropna()
        if len(noise_vars) > 1:
            ax.hist(noise_vars, bins=max(5, len(noise_vars)//3), alpha=0.75, color='#2ca02c', edgecolor='black', linewidth=0.7)
            ax.axvline(noise_vars.mean(), color='#d62728', linestyle='--', linewidth=1.8, alpha=0.8)
            ax.axvline(np.median(noise_vars), color='#17becf', linestyle='--', linewidth=1.8, alpha=0.8)
            ax.set_xlabel('Noise Variance')
            ax.set_ylabel('Count')
            ax.set_title('Noise Variance Distribution', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
        
        # Noise Q-Q Plot vs Normal
        ax = fig.add_subplot(gs[3, 1])
        if len(noise_vars) > 2:
            stats.probplot(noise_vars, dist="norm", plot=ax)
            ax.set_title('Noise Q-Q vs Normal', fontweight='bold', fontsize=12)
            ax.set_xlabel('Normal Quantiles', fontweight='bold')
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
        
        # Noise Q-Q Plot vs Student's t
        ax = fig.add_subplot(gs[3, 2])
        if len(noise_vars) > 3:
            try:
                t_params = stats.t.fit(noise_vars)
                t_df = t_params[0]
                sorted_data = np.sort(noise_vars)
                n = len(sorted_data)
                theoretical_t = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=t_df)
                ax.scatter(theoretical_t, sorted_data, alpha=0.6, s=40, color='#2ca02c')
                
                # Reference line
                slope, intercept = np.polyfit(theoretical_t, sorted_data, 1)
                x_ref = np.array([theoretical_t.min(), theoretical_t.max()])
                ax.plot(x_ref, slope * x_ref + intercept, 'r-', linewidth=1.5, alpha=0.7)
                
                ax.set_xlabel(f"Student's t Quantiles (df={t_df:.1f})", fontweight='bold')
                ax.set_ylabel('Sample Quantiles', fontweight='bold')
                ax.set_title("Noise Q-Q vs Student's t", fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.25)
                ax.set_axisbelow(True)
            except:
                ax.axis('off')
        else:
            ax.axis('off')
    
    clear_figure_titles(fig)
    plt.savefig(output_dir / f'error_distribution_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  - Saved error_distribution_{sensor_name}.png")


def create_offset_gain_correlations(og_df, sensor_name, output_dir=OUTPUT_DIR):
    """
    Create visualizations showing relationships between error sources and predictors.
    Shows correlation scatter plots with regression lines.
    """
    if len(og_df) < 2:
        return
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 15))
    fig.suptitle(f'{sensor_name} - Predictor Relationships', fontsize=14, fontweight='bold', y=0.995)
    
    # ===== ROW 0: TOTAL ERROR vs Predictors =====
    
    # Extract raw errors with predictors
    error_temp_data = []
    error_time_data = []
    
    for idx, row in og_df.iterrows():
        if 'Residuals' in row and isinstance(row['Residuals'], np.ndarray):
            temp = row.get('Temperature', np.nan)
            timespan = row.get('Timespan_seconds', np.nan)
            for i, err in enumerate(row['Residuals']):
                if not np.isnan(temp):
                    error_temp_data.append((temp, err, f'Point {i+1}'))
                if not np.isnan(timespan):
                    error_time_data.append((timespan, err, f'Point {i+1}'))
    
    # Total Error vs Temperature
    ax = axes[0, 0]
    if len(error_temp_data) > 0:
        colors_map = {'Point 1': '#1f77b4', 'Point 2': '#ff7f0e', 'Point 3': '#2ca02c'}
        for point_label in ['Point 1', 'Point 2', 'Point 3']:
            point_data = [(t, e) for t, e, p in error_temp_data if p == point_label]
            if len(point_data) > 0:
                temps, errs = zip(*point_data)
                ax.scatter(temps, errs, alpha=0.6, s=40, color=colors_map[point_label], 
                          label=f'{point_label}, n = {len(point_data)}')
        
        if len(error_temp_data) > 2:
            all_temps = [t for t, e, p in error_temp_data]
            all_errs = [e for t, e, p in error_temp_data]
            z = np.polyfit(all_temps, all_errs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_temps), max(all_temps), 100)
            ax.plot(x_line, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
            r, _ = stats.pearsonr(all_temps, all_errs)
            ax.set_title(f'Total Error vs Temperature\nr = {r:.3f}', fontweight='bold')
        else:
            ax.set_title('Total Error vs Temperature', fontweight='bold')
        
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Total Error', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')
    
    # Total Error vs Timespan
    ax = axes[0, 1]
    if len(error_time_data) > 0:
        colors_map = {'Point 1': '#1f77b4', 'Point 2': '#ff7f0e', 'Point 3': '#2ca02c'}
        for point_label in ['Point 1', 'Point 2', 'Point 3']:
            point_data = [(t, e) for t, e, p in error_time_data if p == point_label]
            if len(point_data) > 0:
                times, errs = zip(*point_data)
                ax.scatter(np.array(times)/86400, errs, alpha=0.6, s=40, color=colors_map[point_label], 
                          label=f'{point_label}, n = {len(point_data)}')
        
        if len(error_time_data) > 2:
            all_times = [t for t, e, p in error_time_data]
            all_errs = [e for t, e, p in error_time_data]
            z = np.polyfit(all_times, all_errs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_times), max(all_times), 100)
            ax.plot(x_line/86400, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
            r, _ = stats.pearsonr(all_times, all_errs)
            ax.set_title(f'Total Error vs Timespan\nr = {r:.3f}', fontweight='bold')
        else:
            ax.set_title('Total Error vs Timespan', fontweight='bold')
        
        ax.set_xlabel('Days Since Calibration', fontweight='bold')
        ax.set_ylabel('Total Error', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')
    
    # Row 1: Offset correlations
    # Offset vs Temperature
    ax = axes[1, 0]
    offset_temp = og_df[['Offset', 'Temperature']].dropna()
    if len(offset_temp) > 2:
        ax.scatter(offset_temp['Temperature'], offset_temp['Offset'], alpha=0.65, s=50, color='#1f77b4')
        z = np.polyfit(offset_temp['Temperature'], offset_temp['Offset'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(offset_temp['Temperature'].min(), offset_temp['Temperature'].max(), 100)
        ax.plot(x_line, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(offset_temp['Temperature'], offset_temp['Offset'])
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Offset', fontweight='bold')
        ax.set_title(f'Offset vs Temperature\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    
    # Offset vs Timespan
    ax = axes[1, 1]
    offset_time = og_df[['Offset', 'Timespan_seconds']].dropna()
    if len(offset_time) > 2:
        ax.scatter(offset_time['Timespan_seconds']/86400, offset_time['Offset'], alpha=0.65, s=50, color='#1f77b4')
        z = np.polyfit(offset_time['Timespan_seconds'], offset_time['Offset'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(offset_time['Timespan_seconds'].min(), offset_time['Timespan_seconds'].max(), 100)
        ax.plot(x_line/86400, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(offset_time['Timespan_seconds'], offset_time['Offset'])
        ax.set_xlabel('Days Since Calibration', fontweight='bold')
        ax.set_ylabel('Offset', fontweight='bold')
        ax.set_title(f'Offset vs Timespan\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    
    # Row 2: Gain correlations
    # Gain vs Temperature (multi-point only)
    ax = axes[2, 0]
    gain_temp = og_df[og_df['Fit_Type'] == 'Multi_Point'][['Gain', 'Temperature']].dropna()
    if len(gain_temp) > 2:
        ax.scatter(gain_temp['Temperature'], gain_temp['Gain'], alpha=0.65, s=50, color='#ff7f0e')
        z = np.polyfit(gain_temp['Temperature'], gain_temp['Gain'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(gain_temp['Temperature'].min(), gain_temp['Temperature'].max(), 100)
        ax.plot(x_line, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(gain_temp['Temperature'], gain_temp['Gain'])
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Gain', fontweight='bold')
        ax.set_title(f'Gain vs Temperature\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')
    
    # Gain vs Timespan (multi-point only)
    ax = axes[2, 1]
    gain_time = og_df[og_df['Fit_Type'] == 'Multi_Point'][['Gain', 'Timespan_seconds']].dropna()
    if len(gain_time) > 2:
        ax.scatter(gain_time['Timespan_seconds'] / 86400, gain_time['Gain'], alpha=0.65, s=50, color='#ff7f0e')
        z = np.polyfit(gain_time['Timespan_seconds'], gain_time['Gain'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(gain_time['Timespan_seconds'].min(), gain_time['Timespan_seconds'].max(), 100)
        ax.plot(x_line / 86400, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(gain_time['Timespan_seconds'], gain_time['Gain'])
        ax.set_xlabel('Days Since Calibration', fontweight='bold')
        ax.set_ylabel('Gain', fontweight='bold')
        ax.set_title(f'Gain vs Timespan\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')

    # Row 3: Noise correlations
    # Noise vs Temperature
    ax = axes[3, 0]
    noise_temp = og_df[['Noise_Variance', 'Temperature']].dropna()
    if len(noise_temp) > 2:
        ax.scatter(noise_temp['Temperature'], noise_temp['Noise_Variance'], alpha=0.65, s=50, color='#2ca02c')
        z = np.polyfit(noise_temp['Temperature'], noise_temp['Noise_Variance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(noise_temp['Temperature'].min(), noise_temp['Temperature'].max(), 100)
        ax.plot(x_line, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(noise_temp['Temperature'], noise_temp['Noise_Variance'])
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Noise Variance', fontweight='bold')
        ax.set_title(f'Noise vs Temperature\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')

    # Noise vs Timespan
    ax = axes[3, 1]
    noise_time = og_df[['Noise_Variance', 'Timespan_seconds']].dropna()
    if len(noise_time) > 2:
        ax.scatter(noise_time['Timespan_seconds'] / 86400, noise_time['Noise_Variance'], alpha=0.65, s=50, color='#2ca02c')
        z = np.polyfit(noise_time['Timespan_seconds'], noise_time['Noise_Variance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(noise_time['Timespan_seconds'].min(), noise_time['Timespan_seconds'].max(), 100)
        ax.plot(x_line / 86400, p(x_line), color='#d62728', linestyle='-', alpha=0.8, linewidth=2)
        r, _ = stats.pearsonr(noise_time['Timespan_seconds'], noise_time['Noise_Variance'])
        ax.set_xlabel('Days Since Calibration', fontweight='bold')
        ax.set_ylabel('Noise Variance', fontweight='bold')
        ax.set_title(f'Noise vs Timespan\nr = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.axis('off')
    
    clear_figure_titles(fig)
    plt.savefig(output_dir / f'error_correlations_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  - Saved error_correlations_{sensor_name}.png")


def create_significance_summary_visualization(og_df, og_stats, sensor_name, output_dir=OUTPUT_DIR):
    """
    Create comprehensive visualization of all significance tests.
    Shows model significance rates, hypothesis test results, and regression significance patterns.
    """
    if len(og_df) < 2:
        return
    
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)
    
    fig.suptitle(f'{sensor_name} - Statistical Significance Summary', fontsize=14, fontweight='bold', y=0.995)
    
    # ===== ROW 1: MODEL SIGNIFICANCE =====
    
    # Panel 1: Model Significance Distribution (Multi-Point Only)
    ax = fig.add_subplot(gs[0, 0])
    multi_df = og_df[og_df['Fit_Type'] == 'Multi_Point'].copy()
    if len(multi_df) > 0:
        sig_counts = multi_df['Model_Significant'].value_counts()
        sig_pct = (multi_df['Model_Significant'].sum() / len(multi_df)) * 100
        colors_sig = ['#d62728', '#2ca02c']
        
        counts = [sig_counts.get(False, 0), sig_counts.get(True, 0)]
        ax.bar([0, 1], counts, color=colors_sig, alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'Model Significance\n({sig_pct:.0f}% significant)', fontweight='bold', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Sig.\n(p≥0.05)', 'Significant\n(p<0.05)'], fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    
    # Panel 2: Model F-statistic Distribution
    ax = fig.add_subplot(gs[0, 1])
    if len(multi_df) > 0:
        f_stats = multi_df['Model_F_stat'].dropna()
        ax.hist(f_stats, bins=max(5, len(f_stats)//3), alpha=0.75, color='#9467bd', edgecolor='black', linewidth=0.7)
        ax.axvline(f_stats.mean(), color='#d62728', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.axvline(np.median(f_stats), color='#1f77b4', linestyle='--', linewidth=1.8, alpha=0.8)
        ax.set_xlabel('F-Statistic', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('F-Statistic Distribution', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    
    # Panel 3: Model p-value Distribution
    ax = fig.add_subplot(gs[0, 2])
    if len(multi_df) > 0:
        p_vals = multi_df['Model_p_value'].dropna()
        neg_log_p = -np.log10(p_vals + 1e-10)
        ax.hist(neg_log_p, bins=max(5, len(p_vals)//3), alpha=0.75, color='#ff7f0e', edgecolor='black', linewidth=0.7)
        ax.axvline(-np.log10(0.05), color='#d62728', linestyle='--', linewidth=2, alpha=0.8)
        ax.set_xlabel('-log₁₀(p-value)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('p-Value Distribution\n(−log₁₀ scale)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    
    # ===== ROW 2: H1 HYPOTHESIS TESTS (Offset/Gain/Noise Stability) =====
    
    # Panel 4: Offset/Gain/Noise Constancy (CV)
    ax = fig.add_subplot(gs[1, 0])
    cv_data = {}
    if 'Offset_CV' in og_stats:
        cv_data['Offset'] = og_stats['Offset_CV']
    if 'Gain_CV' in og_stats:
        cv_data['Gain'] = og_stats['Gain_CV']
    if 'Noise_Variance_CV' in og_stats:
        cv_data['Noise'] = og_stats['Noise_Variance_CV']
    
    if cv_data:
        keys = list(cv_data.keys())
        vals = list(cv_data.values())
        colors_cv = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax.bar(keys, vals, color=colors_cv[:len(keys)], alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.axhline(0.2, color='#d62728', linestyle='--', linewidth=1.8, alpha=0.7)
        ax.set_ylabel('Coefficient of Variation', fontweight='bold')
        ax.set_title('Error Source Stability\n(CV < 0.2 = Stable)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)
    
    # Panel 5: Noise Distribution Tests
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    # Panel 6: Regression Significance Heatmap
    ax = fig.add_subplot(gs[1, 2])
    
    # Build significance matrix
    responses = ['Offset', 'Gain', 'Noise']
    predictors = ['Temperature', 'Timespan']
    sig_matrix = np.full((len(responses), len(predictors)), np.nan)
    
    for i, resp in enumerate(responses):
        for j, pred in enumerate(predictors):
            key = f'{resp}_vs_{pred}_significant'
            if key in og_stats:
                sig_matrix[i, j] = 1 if og_stats[key] else 0
    
    # Create heatmap
    im = ax.imshow(sig_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(predictors)))
    ax.set_yticks(np.arange(len(responses)))
    ax.set_xticklabels(predictors, fontweight='bold')
    ax.set_yticklabels(responses, fontweight='bold')
    ax.set_title('Predictor Significance', fontweight='bold', fontsize=12)
    
    # Add text annotations
    for i in range(len(responses)):
        for j in range(len(predictors)):
            if not np.isnan(sig_matrix[i, j]):
                text = "✓" if sig_matrix[i, j] == 1 else "✗"
                color = 'black'
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=16, fontweight='bold')
    
    # ===== ROW 3: DETAILED REGRESSION RESULTS =====
    
    # Panel 7: Temperature Effects
    ax = fig.add_subplot(gs[2, 0])
    temp_results = {}
    for resp in ['Offset', 'Noise']:
        key_slope = f'{resp}_vs_Temperature_slope'
        key_p = f'{resp}_vs_Temperature_pvalue'
        if key_slope in og_stats and key_p in og_stats:
            p_val = og_stats[key_p]
            sig_marker = "*" if p_val < 0.05 else ""
            temp_results[f'{resp}{sig_marker}'] = og_stats[key_slope]
    
    if temp_results:
        keys = list(temp_results.keys())
        vals = list(temp_results.values())
        colors_temp = ['#1f77b4' if '*' in k else '#e0e0e0' for k in keys]
        ax.barh(keys, vals, color=colors_temp, alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Slope', fontweight='bold')
        ax.set_title('Temperature Effects\n(* p < 0.05)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='x')
        ax.set_axisbelow(True)
    
    # Panel 8: Timespan Effects
    ax = fig.add_subplot(gs[2, 1])
    time_results = {}
    for resp in ['Offset', 'Noise']:
        key_slope = f'{resp}_vs_Timespan_slope'
        key_p = f'{resp}_vs_Timespan_pvalue'
        if key_slope in og_stats and key_p in og_stats:
            p_val = og_stats[key_p]
            sig_marker = "*" if p_val < 0.05 else ""
            time_results[f'{resp}{sig_marker}'] = og_stats[key_slope]
    
    if time_results:
        keys = list(time_results.keys())
        vals = list(time_results.values())
        colors_time = ['#ff7f0e' if '*' in k else '#e0e0e0' for k in keys]
        ax.barh(keys, vals, color=colors_time, alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Slope', fontweight='bold')
        ax.set_title('Timespan Effects\n(* p < 0.05)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.25, axis='x')
        ax.set_axisbelow(True)
    
    # Panel 9: (Reserved)
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    clear_figure_titles(fig)
    plt.savefig(output_dir / f'significance_tests_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  - Saved significance_tests_{sensor_name}.png")


def create_cross_sensor_significance_summary(all_stats, output_dir=OUTPUT_DIR):
    """Create cross-sensor summary of predictor p-values and model fit."""
    if not all_stats:
        return

    stats_df = pd.DataFrame(all_stats)
    if 'Sensor' not in stats_df.columns or stats_df.empty:
        return

    stats_df = stats_df.sort_values('Sensor')
    sensors = stats_df['Sensor'].tolist()

    predictor_keys = [
        ('Offset vs Temp', 'Offset_vs_Temperature'),
        ('Gain vs Temp', 'Gain_vs_Temperature'),
        ('Noise vs Temp', 'Noise_vs_Temperature'),
        ('Offset vs Time', 'Offset_vs_Timespan'),
        ('Gain vs Time', 'Gain_vs_Timespan'),
        ('Noise vs Time', 'Noise_vs_Timespan'),
    ]

    p_robust_matrix = np.full((len(sensors), len(predictor_keys)), np.nan)
    p_ols_matrix = np.full((len(sensors), len(predictor_keys)), np.nan)
    r_matrix = np.full((len(sensors), len(predictor_keys)), np.nan)
    rho_matrix = np.full((len(sensors), len(predictor_keys)), np.nan)
    n_matrix = np.full((len(sensors), len(predictor_keys)), np.nan)
    for i, row in stats_df.reset_index(drop=True).iterrows():
        for j, (_, key_prefix) in enumerate(predictor_keys):
            p_robust_key = f'{key_prefix}_robust_pvalue'
            p_ols_key = f'{key_prefix}_pvalue'
            r_key = f'{key_prefix}_r'
            rho_key = f'{key_prefix}_spearman_r'
            n_key = f'{key_prefix}_n'
            if p_robust_key in row and pd.notna(row[p_robust_key]):
                p_robust_matrix[i, j] = float(row[p_robust_key])
            if p_ols_key in row and pd.notna(row[p_ols_key]):
                p_ols_matrix[i, j] = float(row[p_ols_key])
            if r_key in row and pd.notna(row[r_key]):
                r_matrix[i, j] = float(row[r_key])
            if rho_key in row and pd.notna(row[rho_key]):
                rho_matrix[i, j] = float(row[rho_key])
            if n_key in row and pd.notna(row[n_key]):
                n_matrix[i, j] = float(row[n_key])

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Cross-Sensor Predictor Significance', fontsize=14, fontweight='bold', y=0.995)

    color_matrix = p_robust_matrix if USE_ROBUST_PVALUE_FOR_COLOR else p_ols_matrix
    masked = np.ma.masked_invalid(color_matrix)
    im = ax.imshow(masked, cmap='YlGnBu_r', aspect='auto')
    ax.set_xticks(np.arange(len(predictor_keys)))
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_xticklabels([label for label, _ in predictor_keys], rotation=30, ha='right')
    ax.set_yticklabels(sensors)
    title_suffix = 'Robust' if USE_ROBUST_PVALUE_FOR_COLOR else 'OLS'
    ax.set_title(f'p-value per Predictor ({title_suffix})', fontweight='bold', fontsize=12)

    for i in range(len(sensors)):
        for j in range(len(predictor_keys)):
            p_robust = p_robust_matrix[i, j]
            p_ols = p_ols_matrix[i, j]
            r_val = r_matrix[i, j]
            rho_val = rho_matrix[i, j]
            n_val = n_matrix[i, j]
            if np.isnan(p_robust) and np.isnan(p_ols):
                text = '-'
            else:
                p_robust_text = f"p_rob={p_robust:.2e}" if not np.isnan(p_robust) else ""
                p_ols_text = f"p_ols={p_ols:.2e}" if not np.isnan(p_ols) else ""
                r_text = '' if np.isnan(r_val) else f"r={r_val:.2f}{'*' if abs(r_val) >= PRACTICAL_R_THRESHOLD else ''}"
                rho_text = '' if np.isnan(rho_val) else f"ρ={rho_val:.2f}"
                n_text = '' if np.isnan(n_val) else f"n={int(n_val)}"
                text = f"{p_robust_text} {p_ols_text}\n{r_text} {rho_text}\n{n_text}".strip()
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar_label = 'p-value (robust)' if USE_ROBUST_PVALUE_FOR_COLOR else 'p-value (OLS)'
    cbar.set_label(cbar_label, fontweight='bold')

    clear_figure_titles(fig)
    plt.tight_layout()
    plt.savefig(output_dir / 'significance_summary_across_sensors.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  - Saved significance_summary_across_sensors.png")


def create_distribution_fit_summary(all_distribution_fits, output_dir=OUTPUT_DIR):
    """Create summary visualization of normal vs Student's t fit preference."""
    if not all_distribution_fits:
        return

    dist_df = pd.DataFrame(all_distribution_fits)
    if dist_df.empty or 'Sensor' not in dist_df.columns or 'component' not in dist_df.columns:
        return

    sensors = sorted(dist_df['Sensor'].dropna().unique().tolist())
    components = ['Offset', 'Gain', 'Noise']

    matrix = np.full((len(sensors), len(components)), np.nan)
    annotations = [['' for _ in components] for _ in sensors]

    for i, sensor in enumerate(sensors):
        for j, component in enumerate(components):
            rows = dist_df[(dist_df['Sensor'] == sensor) & (dist_df['component'] == component)]
            if rows.empty:
                continue
            aic_diff = rows.iloc[0].get('aic_diff', np.nan)
            if pd.notna(aic_diff):
                matrix[i, j] = float(aic_diff)
                annotations[i][j] = f"{aic_diff:.1f}"
            else:
                annotations[i][j] = '-'

    if np.all(np.isnan(matrix)):
        return

    max_abs = np.nanmax(np.abs(matrix))
    max_abs = max(max_abs, 1.0)
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Distribution Fit Preference (AIC)', fontsize=14, fontweight='bold', y=0.995)

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_xticklabels(components)
    ax.set_yticklabels(sensors)

    for i in range(len(sensors)):
        for j in range(len(components)):
            ax.text(j, i, annotations[i][j], ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('AIC(normal) - AIC(t)', fontweight='bold')

    ax.set_title('AIC Difference by Component', fontweight='bold', fontsize=12)
    clear_figure_titles(fig)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_fit_summary.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("  - Saved distribution_fit_summary.png")


def create_offset_gain_visualizations(og_df, og_stats, sensor_name, output_dir=OUTPUT_DIR):
    """Create per-sensor offset+gain visualization."""
    if len(og_df) < 1:
        return
    
    # Check if we have multi-point data
    multi_point_df = og_df[og_df['Fit_Type'] == 'Multi_Point']
    has_multi_point = len(multi_point_df) > 0
    
    if not has_multi_point:
        return  # Don't generate figure if no multi-point data
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{sensor_name} - Offset and Gain Analysis', fontsize=14, fontweight='bold', y=0.995)

    # Column 1: Temperature relationships
    # Offset vs Temperature
    ax = axes[0, 0]
    if 'Temperature' in og_df.columns:
        temp_data = og_df[['Offset', 'Temperature']].dropna()
        if len(temp_data) > 1:
            ax.scatter(temp_data['Temperature'], temp_data['Offset'], alpha=0.65, color='#1f77b4', s=50)
            if len(temp_data) > 2:
                z = np.polyfit(temp_data['Temperature'], temp_data['Offset'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(temp_data['Temperature'].min(), temp_data['Temperature'].max(), 100)
                ax.plot(x_line, p(x_line), color='#d62728', linewidth=2, alpha=0.8)
            ax.set_xlabel('Temperature (°C)', fontweight='bold')
            ax.set_ylabel('Offset', fontweight='bold')
            ax.set_title('Offset vs Temperature')
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
    
    # Column 2: Timespan relationships
    # Offset vs Timespan
    ax = axes[0, 1]
    if 'Timespan_seconds' in og_df.columns:
        time_data = og_df[['Offset', 'Timespan_seconds']].dropna()
        if len(time_data) > 1:
            ax.scatter(time_data['Timespan_seconds'] / 86400, time_data['Offset'], alpha=0.65, color='#1f77b4', s=50)
            if len(time_data) > 2:
                z = np.polyfit(time_data['Timespan_seconds'], time_data['Offset'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(time_data['Timespan_seconds'].min(), time_data['Timespan_seconds'].max(), 100)
                ax.plot(x_line / 86400, p(x_line), color='#d62728', linewidth=2, alpha=0.8)
            ax.set_xlabel('Days Since Last Cal', fontweight='bold')
            ax.set_ylabel('Offset', fontweight='bold')
            ax.set_title('Offset vs Timespan')
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
    
    # Gain vs Temperature
    ax = axes[1, 0]
    if len(multi_point_df) > 0:
        gain_temp_data = multi_point_df[['Gain', 'Temperature']].dropna()
        if len(gain_temp_data) > 1:
            ax.scatter(gain_temp_data['Temperature'], gain_temp_data['Gain'], alpha=0.65, color='#ff7f0e', s=50)
            if len(gain_temp_data) > 2:
                z = np.polyfit(gain_temp_data['Temperature'], gain_temp_data['Gain'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(gain_temp_data['Temperature'].min(), gain_temp_data['Temperature'].max(), 100)
                ax.plot(x_line, p(x_line), color='#d62728', linewidth=2, alpha=0.8)
            ax.set_xlabel('Temperature (°C)', fontweight='bold')
            ax.set_ylabel('Gain', fontweight='bold')
            ax.set_title('Gain vs Temperature')
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, 'No Multi-Point Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Gain vs Timespan
    ax = axes[1, 1]
    if len(multi_point_df) > 0:
        gain_time_data = multi_point_df[['Gain', 'Timespan_seconds']].dropna()
        if len(gain_time_data) > 1:
            ax.scatter(gain_time_data['Timespan_seconds'] / 86400, gain_time_data['Gain'], alpha=0.65, color='#ff7f0e', s=50)
            if len(gain_time_data) > 2:
                z = np.polyfit(gain_time_data['Timespan_seconds'], gain_time_data['Gain'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(gain_time_data['Timespan_seconds'].min(), gain_time_data['Timespan_seconds'].max(), 100)
                ax.plot(x_line / 86400, p(x_line), color='#d62728', linewidth=2, alpha=0.8)
            ax.set_xlabel('Days Since Last Cal', fontweight='bold')
            ax.set_ylabel('Gain', fontweight='bold')
            ax.set_title('Gain vs Timespan')
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, 'No Multi-Point Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Column 3: Model quality
    # R² distribution
    ax = axes[0, 2]
    if 'R_squared' in og_df.columns:
        r2_data = og_df[og_df['Fit_Type'] == 'Multi_Point']['R_squared'].dropna()
        if len(r2_data) > 0:
            ax.hist(r2_data, bins=10, alpha=0.75, color='#2ca02c', edgecolor='black', linewidth=0.7)
            ax.set_xlabel('R²', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Model Fit Quality (N={len(r2_data)})')
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_axisbelow(True)
        else:
            ax.axis('off')
    else:
        ax.axis('off')
    
    # Fit type distribution
    ax = axes[1, 2]
    fit_counts = og_df['Fit_Type'].value_counts()
    colors = ['#1f77b4', '#ff7f0e']
    ax.pie(fit_counts.values, labels=fit_counts.index, autopct='%1.1f%%', colors=colors[:len(fit_counts)])
    ax.set_title('Calibration Point Distribution')
    
    clear_figure_titles(fig)
    plt.tight_layout()
    plt.savefig(output_dir / f'offset_gain_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  - Saved offset_gain_{sensor_name}.png")


def main():
    """Main execution."""
    print(f"Loading correction files from: {CORRECTIONS_DIR}\n")
    
    all_csv_files = sorted(CORRECTIONS_DIR.glob("*.csv"))
    skip_prefixes = ['alternative_distributions', 'correlation_statistics', 'distribution_statistics',
                     'independence_tests', 'normality_statistics', 'offset_gain', 'separate_point']
    
    filtered_csv_files = [f for f in all_csv_files 
                         if not any(f.stem.startswith(skip) for skip in skip_prefixes)]
    
    sensor_files = {}
    excluded_count = 0
    
    for csv_file in filtered_csv_files:
        normalized_name = normalize_sensor_name(csv_file)
        if normalized_name is None:
            print(f"Excluding: {csv_file.stem}")
            excluded_count += 1
            continue
        
        if normalized_name in sensor_files:
            if '(µS_cm)' in csv_file.name or 'microS' in csv_file.name:
                sensor_files[normalized_name] = csv_file
        else:
            sensor_files[normalized_name] = csv_file
    
    if excluded_count > 0:
        print(f"Found and excluded {excluded_count} duplicate/invalid sensor file(s)\n")
    
    if not sensor_files:
        print("No valid sensors found!")
        return
    
    print(f"Found {len(sensor_files)} unique sensor(s) to analyze\n")
    
    aggregate_dir = OUTPUT_DIR / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Storage for all results
    all_offset_gain_results = []
    all_offset_gain_stats = []
    all_separate_stats = []
    all_distribution_fits = []
    
    # Process each sensor
    for sensor_name, csv_file in sorted(sensor_files.items()):
        print(f"Processing: {sensor_name}")
        
        try:
            sensor_output_dir = OUTPUT_DIR / sensor_name
            sensor_output_dir.mkdir(parents=True, exist_ok=True)
            raw_df = load_correction_data(csv_file)
            print(f"  - Loaded {len(raw_df)} calibration events")
            
            # Apply filtering logic
            raw_df = apply_filtering_logic(raw_df)
            print(f"  - After filtering: {len(raw_df)} calibration events")
            
            # ===== OFFSET+GAIN ANALYSIS =====
            print(f"  - Fitting offset+gain models...")
            og_df = calculate_offset_gain_results(raw_df, sensor_name)
            
            if len(og_df) > 0:
                og_df['Sensor'] = sensor_name
                all_offset_gain_results.append(og_df)
                
                # Test hypotheses
                og_stats = test_offset_gain_statistics(og_df, sensor_name)
                all_offset_gain_stats.append(og_stats)
                print(f"    - Offset+Gain: {len(og_df)} events fitted")
                
                # Test distribution fits (Normal vs Student's t)
                print(f"    - Testing distribution fits...")
                offset_fit = test_distribution_fit(og_df['Offset'].dropna().values, 'Offset')
                offset_fit['Sensor'] = sensor_name
                all_distribution_fits.append(offset_fit)
                
                gains = og_df[og_df['Fit_Type'] == 'Multi_Point']['Gain'].dropna().values
                if len(gains) >= 3:
                    gain_fit = test_distribution_fit(gains, 'Gain')
                    gain_fit['Sensor'] = sensor_name
                    all_distribution_fits.append(gain_fit)
                
                noise_fit = test_distribution_fit(og_df['Noise_Variance'].dropna().values, 'Noise')
                noise_fit['Sensor'] = sensor_name
                all_distribution_fits.append(noise_fit)
                
                # Visualizations
                create_error_distribution_visualizations(og_df, sensor_name, output_dir=sensor_output_dir)
                create_offset_gain_correlations(og_df, sensor_name, output_dir=sensor_output_dir)
                create_offset_gain_visualizations(og_df, og_stats, sensor_name, output_dir=sensor_output_dir)
            
            # ===== SEPARATE POINT ANALYSIS =====
            print(f"  - Analyzing separate calibration points...")
            separate_data = prepare_separate_analysis(raw_df, sensor_name)
            
            seen_points = set()
            for point_name, df in separate_data.items():
                if point_name in seen_points:
                    continue
                seen_points.add(point_name)
                if len(df) > 0:
                    point_stats = analyze_separate_point(df, sensor_name, point_name)
                    point_stats['Sensor'] = sensor_name
                    point_stats['Calibration_Point'] = point_name
                    all_separate_stats.append(point_stats)
                    print(f"    - {point_name}: {len(df)} events")
            
            print()
        
        except Exception as e:
            print(f"  [ERROR] {str(e)}\n")
            continue
    
    # ===== SAVE RESULTS =====
    print("\nSaving results...")
    
    if all_offset_gain_results:
        og_combined = pd.concat(all_offset_gain_results, ignore_index=True)
        og_combined.to_csv(aggregate_dir / 'offset_gain_model_results.csv', index=False)
        print(f"  - Saved offset_gain_model_results.csv ({len(og_combined)} events)")
    
    if all_offset_gain_stats:
        og_stats_df = pd.DataFrame(all_offset_gain_stats)
        og_stats_df.to_csv(aggregate_dir / 'offset_gain_statistics.csv', index=False)
        print(f"  - Saved offset_gain_statistics.csv")

        create_cross_sensor_significance_summary(all_offset_gain_stats, output_dir=aggregate_dir)
    
    if all_separate_stats:
        sep_df = pd.DataFrame(all_separate_stats)
        sep_df.to_csv(aggregate_dir / 'separate_point_statistics.csv', index=False)
        print(f"  - Saved separate_point_statistics.csv ({len(sep_df)} calibration points)")
    
    if all_distribution_fits:
        dist_df = pd.DataFrame(all_distribution_fits)
        dist_df.to_csv(aggregate_dir / 'distribution_goodness_of_fit.csv', index=False)
        print(f"  - Saved distribution_goodness_of_fit.csv ({len(dist_df)} components)")
        create_distribution_fit_summary(all_distribution_fits, output_dir=aggregate_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Aggregate results saved to: {aggregate_dir}")


if __name__ == "__main__":
    main()

