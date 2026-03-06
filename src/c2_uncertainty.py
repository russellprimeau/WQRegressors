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
    # Note: Cannot estimate variance from single observation
    if len(post_cals) == 1:
        return {
            'Offset': errors[0],
            'Gain': 0.0,
            'Noise_Variance': np.nan,
            'Noise_Std': np.nan,
            'Model_F_stat': np.nan,
            'Model_p_value': np.nan,
            'Model_Significant': False,
            'N_Parameters': 1,
            'Fit_Type': 'Single_Point',
            'N_Points': 1,
            'Residuals': errors,
            'PostCal_Values': post_cals,
            'Error_Values': errors
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
            'Residuals': residuals,
            'PostCal_Values': post_cals,
            'Error_Values': errors
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


def test_corrections_kruskal_wallis(raw_df, sensor_name):
    """
    Test whether corrections across calibration points are drawn from the same distribution.
    Performs both Kruskal-Wallis (non-parametric) and one-way ANOVA (parametric).
    Also runs Levene's test for equality of variances and Tukey HSD post-hoc if ANOVA is significant.
    
    Requirements:
    - At least 2 calibration points
    - Each point must have at least 5 valid observations
    - Filter by Stability Achieved == "Yes"
    """
    # Identify correction columns and their corresponding stability columns
    correction_cols = [col for col in raw_df.columns if col.startswith('Correction')]
    stability_cols = []
    
    for col in correction_cols:
        if col == 'Correction1':
            stability_cols.append('Stability Achieved')
        else:
            point_num = col.replace('Correction', '')
            stability_cols.append(f'Stability Achieved {point_num}')
    
    # Extract valid data for each calibration point
    point_groups = {}
    for corr_col, stab_col in zip(correction_cols, stability_cols):
        if corr_col in raw_df.columns and stab_col in raw_df.columns:
            valid_data = raw_df[
                (raw_df[stab_col] == 'Yes') & 
                (raw_df[corr_col].notna())
            ][corr_col].dropna().values
            
            if len(valid_data) >= 5:
                point_groups[corr_col] = valid_data
    
    # Need at least 2 points with sufficient data
    if len(point_groups) < 2:
        return {
            'Sensor': sensor_name,
            'Calibration_Point': 'Across All Points',
            'N_Points_Tested': len(point_groups),
            'Kruskal_Wallis_H': np.nan,
            'Kruskal_Wallis_p': np.nan,
            'Kruskal_Wallis_Significant': False,
            'ANOVA_F': np.nan,
            'ANOVA_p': np.nan,
            'ANOVA_Significant': False,
            'Eta_Squared': np.nan,
            'Levene_F': np.nan,
            'Levene_p': np.nan,
            'Levene_Significant': False,
            'Tukey_HSD_Result': 'N/A (insufficient points)',
            'Test_Status': 'Insufficient data'
        }
    
    # Prepare data for tests
    groups_list = [point_groups[key] for key in sorted(point_groups.keys())]
    group_labels = sorted(point_groups.keys())
    
    # Calculate descriptive statistics
    medians = [np.median(g) for g in groups_list]
    means = [np.mean(g) for g in groups_list]
    stds = [np.std(g) for g in groups_list]
    sample_sizes = [len(g) for g in groups_list]
    
    # 1. Kruskal-Wallis Test (non-parametric)
    kw_stat, kw_p = stats.kruskal(*groups_list)
    kw_significant = kw_p < 0.05
    
    # 2. Levene's Test for equality of variances
    levene_stat, levene_p = stats.levene(*groups_list)
    levene_significant = levene_p < 0.05
    
    # 3. One-way ANOVA (parametric)
    anova_f, anova_p = stats.f_oneway(*groups_list)
    anova_significant = anova_p < 0.05
    
    # 4. Calculate Eta-Squared (effect size for ANOVA)
    all_data = np.concatenate(groups_list)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_list)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    
    # 5. Tukey HSD Post-hoc test (if ANOVA is significant)
    tukey_result = "Not significant (no post-hoc needed)"
    if anova_significant:
        try:
            from scipy.stats import tukey_hsd
            res = tukey_hsd(*groups_list)
            tukey_result = f"Pairwise p-values matrix computed; see visualization"
        except:
            # Fallback: manual pairwise t-tests with Bonferroni correction
            n_comparisons = len(groups_list) * (len(groups_list) - 1) // 2
            bonferroni_alpha = 0.05 / n_comparisons
            pairwise_results = []
            
            for i in range(len(groups_list)):
                for j in range(i + 1, len(groups_list)):
                    t_stat, t_p = stats.ttest_ind(groups_list[i], groups_list[j])
                    sig = "**" if t_p < bonferroni_alpha else ""
                    pairwise_results.append(
                        f"{group_labels[i]} vs {group_labels[j]}: p={t_p:.4f} {sig}"
                    )
            tukey_result = "; ".join(pairwise_results)
    
    return {
        'Sensor': sensor_name,
        'Calibration_Point': 'Across All Points',
        'N_Points_Tested': len(point_groups),
        'Sample_Sizes': str(sample_sizes),
        'Medians': str([f'{m:.4f}' for m in medians]),
        'Means': str([f'{m:.4f}' for m in means]),
        'Std_Devs': str([f'{s:.4f}' for s in stds]),
        'Kruskal_Wallis_H': kw_stat,
        'Kruskal_Wallis_p': kw_p,
        'Kruskal_Wallis_Significant': kw_significant,
        'ANOVA_F': anova_f,
        'ANOVA_p': anova_p,
        'ANOVA_Significant': anova_significant,
        'Eta_Squared': eta_squared,
        'Levene_F': levene_stat,
        'Levene_p': levene_p,
        'Levene_Significant': levene_significant,
        'Tukey_HSD_Result': tukey_result,
        'Test_Status': 'Completed'
    }


def create_corrections_comparison_visualization(raw_df, sensor_name, output_dir=OUTPUT_DIR):
    """
    Create separate box plot and statistical test results visualizations.
    - Box plot: Publication-ready figure with minimal white space, no title
    - Test results: Separate text-only figure with statistical results
    """
    # Identify correction columns and their corresponding stability columns
    correction_cols = [col for col in raw_df.columns if col.startswith('Correction')]
    stability_cols = []
    
    for col in correction_cols:
        if col == 'Correction1':
            stability_cols.append('Stability Achieved')
        else:
            point_num = col.replace('Correction', '')
            stability_cols.append(f'Stability Achieved {point_num}')
    
    # Extract valid data for each calibration point
    point_groups = {}
    for corr_col, stab_col in zip(correction_cols, stability_cols):
        if corr_col in raw_df.columns and stab_col in raw_df.columns:
            valid_data = raw_df[
                (raw_df[stab_col] == 'Yes') & 
                (raw_df[corr_col].notna())
            ][corr_col].dropna().values
            
            if len(valid_data) >= 5:
                point_groups[corr_col] = valid_data
    
    # Need at least 2 points
    if len(point_groups) < 2:
        return
    
    # Run tests
    test_results = test_corrections_kruskal_wallis(raw_df, sensor_name)
    
    # Prepare data for visualization
    groups_list = [point_groups[key] for key in sorted(point_groups.keys())]
    group_labels = sorted(point_groups.keys())
    
    # ===== FIGURE 1: BOX PLOT (Publication-ready) =====
    fig_bp, ax_bp = plt.subplots(figsize=(7, 5))
    
    bp = ax_bp.boxplot(groups_list, labels=group_labels, patch_artist=True, widths=0.6)
    
    # Color boxes and add mean markers
    for patch, group in zip(bp['boxes'], groups_list):
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for i, group in enumerate(groups_list):
        mean_val = np.mean(group)
        ax_bp.plot(i + 1, mean_val, 'D', color='green', markersize=8, 
                   label='Mean' if i == 0 else '', zorder=3)
        ax_bp.text(i + 1, mean_val, f'  {mean_val:.2f}', fontsize=9, va='center')
        # Add sample size below x-axis
        ax_bp.text(i + 1, ax_bp.get_ylim()[0], f'n={len(group)}', 
                   ha='center', fontsize=9, color='darkblue', weight='bold')
    
    ax_bp.set_ylabel('Correction Value', fontsize=12, weight='bold')
    ax_bp.set_xlabel('Calibration Point', fontsize=12, weight='bold')
    ax_bp.grid(True, alpha=0.3, axis='y', which='both')
    ax_bp.legend(loc='best', fontsize=10)
    
    # No title on the box plot for publication
    
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
    
    # Save box plot
    output_path_bp = Path(output_dir) / f"{sensor_name}_corrections_boxplot.png"
    output_path_bp.parent.mkdir(parents=True, exist_ok=True)
    fig_bp.savefig(output_path_bp, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig_bp)
    
    # ===== FIGURE 2: TEST RESULTS (Text-only) =====
    fig_text, ax_text = plt.subplots(figsize=(9, 8))
    ax_text.axis('off')
    
    # Format test results
    kw_sig_text = "✓ Significant" if test_results['Kruskal_Wallis_Significant'] else "✗ Not Significant"
    anova_sig_text = "✓ Significant" if test_results['ANOVA_Significant'] else "✗ Not Significant"
    levene_sig_text = "✓ Significant (variances differ)" if test_results['Levene_Significant'] else "✗ Not Significant (equal variances)"
    
    summary_text = f"""
STATISTICAL TEST RESULTS: {sensor_name}

Kruskal-Wallis Test (Non-parametric)
  H-statistic: {test_results['Kruskal_Wallis_H']:.4f}
  p-value: {test_results['Kruskal_Wallis_p']:.4f}
  Result: {kw_sig_text}
  Tests: Whether distributions differ across points

One-way ANOVA (Parametric)
  F-statistic: {test_results['ANOVA_F']:.4f}
  p-value: {test_results['ANOVA_p']:.4f}
  Result: {anova_sig_text}
  Effect Size (η²): {test_results['Eta_Squared']:.4f}
  Tests: Whether means differ across points

Levene's Test (Variance Equality)
  F-statistic: {test_results['Levene_F']:.4f}
  p-value: {test_results['Levene_p']:.4f}
  Result: {levene_sig_text}
  Tests: Whether variances are equal across points

Tukey HSD Post-hoc Comparison:
  {test_results['Tukey_HSD_Result']}

INTERPRETATION:
• If ANOVA significant but Kruskal-Wallis not:
  Mean differences detected despite similar distributions
• If both significant: Distributions differ systematically
• Levene's violation suggests caution with ANOVA
    """
    
    ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save test results
    output_path_text = Path(output_dir) / f"{sensor_name}_corrections_statistics.png"
    fig_text.savefig(output_path_text, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig_text)


def save_sensor_uncertainty_summary(og_df, sensor_name, n_calibration_points, output_dir=OUTPUT_DIR):
    """
    Save sensor uncertainty distribution parameters for Monte Carlo sampling in d_Resample.py.
    
    Outputs a CSV with:
    - Number of calibration points available
    - Offset distribution (mean, std, preferred distribution type)
    - Offset Student's t fit parameters (df, loc, scale)
    - Gain distribution (if >=2 calibration points)
    - Noise distribution (if >=3 calibration points)
    """
    summary = {
        'Sensor': sensor_name,
        'N_Calibration_Points': n_calibration_points,
    }
    
    # Offset (always available)
    offsets = og_df['Offset'].dropna().values
    if len(offsets) > 0:
        summary['Offset_Mean'] = np.mean(offsets)
        summary['Offset_Std'] = np.std(offsets)
        # Try to determine preferred distribution from AIC
        offset_fit = test_distribution_fit(offsets, 'Offset')
        preferred = offset_fit.get('preferred', 'normal')
        summary['Offset_Distribution'] = preferred if preferred != 'equivalent' else 'normal'
        summary['Offset_t_df'] = offset_fit.get('t_df', np.nan)
        summary['Offset_t_loc'] = offset_fit.get('t_loc', np.nan)
        summary['Offset_t_scale'] = offset_fit.get('t_scale', np.nan)
    else:
        summary['Offset_Mean'] = np.nan
        summary['Offset_Std'] = np.nan
        summary['Offset_Distribution'] = 'normal'
        summary['Offset_t_df'] = np.nan
        summary['Offset_t_loc'] = np.nan
        summary['Offset_t_scale'] = np.nan
    
    # Gain (if >=2 calibration points)
    if n_calibration_points >= 2:
        gains = og_df[og_df['Fit_Type'] == 'Multi_Point']['Gain'].dropna().values
        if len(gains) > 0:
            summary['Gain_Mean'] = np.mean(gains)
            summary['Gain_Std'] = np.std(gains)
            gain_fit = test_distribution_fit(gains, 'Gain')
            preferred = gain_fit.get('preferred', 'normal')
            summary['Gain_Distribution'] = preferred if preferred != 'equivalent' else 'normal'
        else:
            summary['Gain_Mean'] = np.nan
            summary['Gain_Std'] = np.nan
            summary['Gain_Distribution'] = 'normal'
    else:
        summary['Gain_Mean'] = np.nan
        summary['Gain_Std'] = np.nan
        summary['Gain_Distribution'] = 'normal'
    
    # Noise (if >=3 calibration points)
    if n_calibration_points >= 3:
        noise_vars = og_df['Noise_Variance'].dropna().values
        if len(noise_vars) > 0:
            summary['Noise_Variance_Mean'] = np.mean(noise_vars)
            summary['Noise_Variance_Std'] = np.std(noise_vars)
            summary['Noise_Std_Mean'] = np.sqrt(np.mean(noise_vars))  # For convenience
            summary['Noise_Std_Std'] = np.std(np.sqrt(noise_vars))
            noise_fit = test_distribution_fit(noise_vars, 'Noise')
            preferred = noise_fit.get('preferred', 'normal')
            summary['Noise_Distribution'] = preferred if preferred != 'equivalent' else 'normal'
        else:
            summary['Noise_Variance_Mean'] = np.nan
            summary['Noise_Variance_Std'] = np.nan
            summary['Noise_Std_Mean'] = np.nan
            summary['Noise_Std_Std'] = np.nan
            summary['Noise_Distribution'] = 'normal'
    else:
        summary['Noise_Variance_Mean'] = np.nan
        summary['Noise_Variance_Std'] = np.nan
        summary['Noise_Std_Mean'] = np.nan
        summary['Noise_Std_Std'] = np.nan
        summary['Noise_Distribution'] = 'normal'
    
    # Save to CSV in sensor-specific folder
    sensor_output_dir = output_dir / sensor_name
    sensor_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = sensor_output_dir / f'{sensor_name}_uncertainty_summary.csv'
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved uncertainty summary: {output_path}")
    
    return summary


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
    
    # Check if we have multi-point data (Gain row)
    has_multi_point = (og_df['Fit_Type'] == 'Multi_Point').any()
    n_rows = 4 if has_multi_point else 3
    fig_height = n_rows * 3  # 3 inches per row
    noise_row = 3 if has_multi_point else 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, fig_height))
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
    
    # Row 2: Gain correlations (only if multi-point data exists)
    if has_multi_point:
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

    # Noise correlations (row 2 if no Gain, row 3 if Gain exists)
    # Noise vs Temperature
    ax = axes[noise_row, 0]
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
    ax = axes[noise_row, 1]
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
    
    clear_figure_titles(fig)
    plt.tight_layout()
    plt.savefig(output_dir / f'offset_gain_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  - Saved offset_gain_{sensor_name}.png")
    
    # Create separate R² histogram if informative (variance exists)
    if 'R_squared' in og_df.columns:
        r2_data = multi_point_df['R_squared'].dropna()
        if len(r2_data) > 2 and r2_data.std() > 0.01:  # Skip if nearly constant
            fig_r2, ax_r2 = plt.subplots(1, 1, figsize=(8, 6))
            ax_r2.hist(r2_data, bins=max(5, len(r2_data)//3), alpha=0.75, color='#2ca02c', edgecolor='black', linewidth=0.7)
            ax_r2.set_xlabel('R²', fontweight='bold')
            ax_r2.set_ylabel('Frequency', fontweight='bold')
            ax_r2.set_title(f'Model Fit Quality (N={len(r2_data)})')
            ax_r2.grid(True, alpha=0.25, axis='y')
            ax_r2.set_axisbelow(True)
            
            clear_figure_titles(fig_r2)
            plt.tight_layout()
            plt.savefig(output_dir / f'r_squared_distribution_{sensor_name}.png', dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  - Saved r_squared_distribution_{sensor_name}.png")


def compare_decomposed_vs_simple_model(results_df, raw_df, sensor_name):
    """
    Compare decomposed model vs simple model using Leave-One-Out Cross-Validation.
    
    Simple Model: Each calibration point has independent N(μⱼ, σⱼ) distribution
                  Correction_j ~ N(μⱼ, σⱼ)
    
    Decomposed Model: Hierarchical model where for each event i:
                      Offset_i ~ N(μ_off, σ_off)
                      Gain_i ~ N(μ_gain, σ_gain)
                      Correction_ji = Offset_i + Gain_i × PostCal_ji + ε_ji
                      where ε_ji ~ N(0, σ_noise)
                      
                      For prediction at new correction with PostCal value pc:
                      E[Correction] = μ_off + μ_gain × pc
                      Var(Correction) = σ_off² + pc² × σ_gain² + σ_noise²
    
    Uses LOOCV: For each event, fit both models on all other events,
    then evaluate predictive likelihood on held-out event's corrections.
    
    Compares via cross-validated log-likelihood and AIC.
    
    Returns dict with test results and recommendation.
    """
    # Initialize results
    comparison = {
        'Sensor': sensor_name,
        'N_Events': len(results_df),
        'N_Corrections': 0,
        'Simple_LogLik': np.nan,
        'Simple_K': np.nan,
        'Simple_AIC': np.nan,
        'Decomposed_LogLik': np.nan,
        'Decomposed_K': np.nan,
        'Decomposed_AIC': np.nan,
        'Delta_AIC': np.nan,
        'Recommendation': 'Insufficient_Data',
        'Reasons': 'Not enough data'
    }
    
    # Collect all corrections organized by calibration point column position
    # Structure: {point_index: [(correction_value, postcal_value)]}
    corrections_by_point = {}
    
    # Extract from results_df which has PostCal_Values and Error_Values
    for idx, row in results_df.iterrows():
        if 'PostCal_Values' not in row or 'Error_Values' not in row:
            continue
        
        postcals = row['PostCal_Values']
        errors = row['Error_Values']
        
        if not isinstance(postcals, np.ndarray) or not isinstance(errors, np.ndarray):
            continue
        
        # Each event has 1-3 points; assign to point index based on order
        for point_idx, (pc, err) in enumerate(zip(postcals, errors), start=1):
            if point_idx not in corrections_by_point:
                corrections_by_point[point_idx] = []
            corrections_by_point[point_idx].append((err, pc))
    
    # Need at least some data
    if not corrections_by_point:
        print(f"    - WARNING: No correction data available")
        return comparison
    
    n_points = len(corrections_by_point)
    n_corrections = sum(len(v) for v in corrections_by_point.values())
    comparison['N_Corrections'] = n_corrections
    
    print(f"    - Found {n_corrections} corrections across {n_points} calibration point(s): {list(corrections_by_point.keys())}")
    for point_idx, data in corrections_by_point.items():
        print(f"      Point {point_idx}: {len(data)} corrections")
    
    # Reset index to ensure consecutive integers for LOOCV
    results_df = results_df.reset_index(drop=True)
    
    # ===== LEAVE-ONE-OUT CROSS-VALIDATION =====
    print(f"\n    - Performing Leave-One-Out Cross-Validation (n={len(results_df)} events)...")
    
    # Storage for predictions and errors
    simple_predictions = []  # (actual, predicted, std, point_idx)
    decomposed_predictions = []  # (actual, predicted, std, point_idx)
    
    # Diagnostic storage
    decomposed_component_variances = []  # Track σ_off, σ_gain, σ_noise per fold
    
    simple_cv_loglik = 0.0
    decomposed_cv_loglik = 0.0
    
    cv_used = 0
    cv_skipped = 0
    # For each event, hold it out and fit models on remaining events
    for holdout_idx in range(len(results_df)):
        holdout_event = results_df.iloc[holdout_idx]
        train_df = results_df.drop(holdout_idx).reset_index(drop=True)
        
        # Get holdout event's corrections
        if 'PostCal_Values' not in holdout_event or 'Error_Values' not in holdout_event:
            continue
        holdout_postcals = holdout_event['PostCal_Values']
        holdout_errors = holdout_event['Error_Values']
        if not isinstance(holdout_postcals, np.ndarray) or not isinstance(holdout_errors, np.ndarray):
            continue
        
        # Determine holdout event's calibration point count
        n_holdout_points = len(holdout_postcals)
        
        # Get holdout event's fitted parameters
        holdout_offset = holdout_event['Offset']
        holdout_gain = holdout_event['Gain']
        holdout_residuals = holdout_event['Residuals'] if 'Residuals' in holdout_event else np.array([])
        holdout_fit_type = holdout_event['Fit_Type']
        
        # ===== DECOMPOSED MODEL (trained on remaining events) =====
        # Model definition depends on calibration point count:
        # 1-point: σ_total only (all variation lumped together)
        # 2-point: σ_gain + σ_residual (gain variation + combined offset/noise)
        # 3+ point: σ_offset + σ_gain + σ_noise (all three components separable)
        
        if n_holdout_points == 1:
            # 1-POINT CASE: Estimate total variance from single-point corrections
            train_single_point = train_df[train_df['N_Points'] == 1]
            if len(train_single_point) < 2:
                cv_skipped += 1
                continue
            
            # Collect all single-point corrections
            train_corrections_1pt = []
            for idx, row in train_single_point.iterrows():
                if 'Error_Values' in row and isinstance(row['Error_Values'], np.ndarray):
                    train_corrections_1pt.extend(row['Error_Values'])
            
            if len(train_corrections_1pt) < 2:
                cv_skipped += 1
                continue
            
            mu_total = np.mean(train_corrections_1pt)
            sigma_total = np.std(train_corrections_1pt, ddof=1)
            
            if sigma_total == 0.0:
                sigma_total = 1e-10
            
            # Covariance: σ²_total × I (1×1 diagonal)
            mean_vec = np.array([mu_total])
            cov = np.array([[sigma_total**2 + 1e-12]])
            
            decomposed_component_variances.append({
                'sigma_total': sigma_total,
                'sigma_offset': np.nan,
                'sigma_gain': np.nan,
                'sigma_noise': np.nan,
                'n_train_events': len(train_corrections_1pt),
                'case': '1-point'
            })
            
        elif n_holdout_points == 2:
            # 2-POINT CASE: Estimate gain variance and residual variance
            train_two_point = train_df[train_df['N_Points'] == 2]
            if len(train_two_point) < 2:
                cv_skipped += 1
                continue
            
            # Estimate gain distribution
            train_gains_2pt = train_two_point['Gain'].dropna().values
            if len(train_gains_2pt) < 2:
                cv_skipped += 1
                continue
            
            mu_gain = np.mean(train_gains_2pt)
            sigma_gain = np.std(train_gains_2pt, ddof=1)
            
            if sigma_gain == 0.0:
                sigma_gain = 1e-10
            
            # Estimate residual variance (offset + noise combined)
            # For 2-point, residual = actual_correction - gain*postcal
            train_residuals_2pt = []
            for idx, row in train_two_point.iterrows():
                if 'Error_Values' in row and 'PostCal_Values' in row and 'Gain' in row:
                    errors = row['Error_Values']
                    postcals = row['PostCal_Values']
                    gain = row['Gain']
                    if isinstance(errors, np.ndarray) and isinstance(postcals, np.ndarray):
                        # Residual from gain-only model (no offset estimated)
                        residuals = errors - gain * postcals
                        train_residuals_2pt.extend(residuals)
            
            if len(train_residuals_2pt) < 2:
                cv_skipped += 1
                continue
            
            sigma_residual = np.std(train_residuals_2pt, ddof=1)
            
            if sigma_residual == 0.0:
                sigma_residual = 1e-10
            
            # Covariance: σ²_gain × (pc ⊗ pc) + σ²_residual × I (2×2)
            mean_vec = mu_gain * holdout_postcals
            pc_outer = np.outer(holdout_postcals, holdout_postcals)
            cov = (sigma_gain**2) * pc_outer + (sigma_residual**2) * np.eye(2)
            cov += np.eye(2) * 1e-12
            
            decomposed_component_variances.append({
                'sigma_total': np.nan,
                'sigma_offset': np.nan,
                'sigma_gain': sigma_gain,
                'sigma_residual': sigma_residual,
                'sigma_noise': np.nan,
                'n_train_events': len(train_gains_2pt),
                'n_residuals': len(train_residuals_2pt),
                'case': '2-point'
            })
            
        else:  # n_holdout_points >= 3
            # 3+ POINT CASE: All three components identifiable
            train_multi_point = train_df[train_df['N_Points'] >= 3]
            if len(train_multi_point) < 2:
                cv_skipped += 1
                continue
            
            # Estimate offset distribution
            train_offsets = train_multi_point['Offset'].dropna().values
            if len(train_offsets) < 2:
                cv_skipped += 1
                continue
            
            mu_offset = np.mean(train_offsets)
            sigma_offset = np.std(train_offsets, ddof=1)
            
            # Estimate gain distribution
            train_gains = train_multi_point['Gain'].dropna().values
            if len(train_gains) < 2:
                cv_skipped += 1
                continue
            
            mu_gain = np.mean(train_gains)
            sigma_gain = np.std(train_gains, ddof=1)
            
            # Estimate noise from residuals
            train_residuals = []
            for idx, row in train_multi_point.iterrows():
                if 'Residuals' in row and isinstance(row['Residuals'], np.ndarray):
                    if len(row['Residuals']) > 0:
                        train_residuals.extend(row['Residuals'])
            
            if len(train_residuals) < 2:
                cv_skipped += 1
                continue
            
            sigma_noise = np.std(train_residuals, ddof=1)
            
            # Handle zero variance
            if sigma_offset == 0.0:
                sigma_offset = 1e-10
            if sigma_gain == 0.0:
                sigma_gain = 1e-10
            if sigma_noise == 0.0:
                sigma_noise = 1e-10
            
            # Covariance: σ²_offset × 11^T + σ²_gain × (pc ⊗ pc) + σ²_noise × I
            mean_vec = mu_offset + mu_gain * holdout_postcals
            ones = np.ones((n_holdout_points, n_holdout_points))
            pc_outer = np.outer(holdout_postcals, holdout_postcals)
            cov = (sigma_offset**2) * ones + (sigma_gain**2) * pc_outer + (sigma_noise**2) * np.eye(n_holdout_points)
            cov += np.eye(n_holdout_points) * 1e-12
            
            decomposed_component_variances.append({
                'sigma_total': np.nan,
                'sigma_offset': sigma_offset,
                'sigma_gain': sigma_gain,
                'sigma_residual': np.nan,
                'sigma_noise': sigma_noise,
                'n_train_events': len(train_offsets),
                'n_residuals': len(train_residuals),
                'case': '3+ point'
            })
            
        # ===== SIMPLE MODEL (trained on remaining events) =====
        # Build point-specific distributions from training data
        train_by_point = {}
        for idx, row in train_df.iterrows():
            if 'PostCal_Values' not in row or 'Error_Values' not in row:
                continue
            postcals = row['PostCal_Values']
            errors = row['Error_Values']
            if not isinstance(postcals, np.ndarray) or not isinstance(errors, np.ndarray):
                continue
            for point_idx, (pc, err) in enumerate(zip(postcals, errors), start=1):
                if point_idx not in train_by_point:
                    train_by_point[point_idx] = []
                train_by_point[point_idx].append(err)
        
        # Predict each correction in holdout event
        for point_idx, (pc, actual_err) in enumerate(zip(holdout_postcals, holdout_errors), start=1):
            if point_idx in train_by_point and len(train_by_point[point_idx]) >= 2:
                train_corrections = np.array(train_by_point[point_idx])
                mu_simple = np.mean(train_corrections)
                sigma_simple = np.std(train_corrections, ddof=1)
                
                # Handle exactly zero variance (all corrections identical)
                if sigma_simple == 0.0:
                    sigma_simple = 1e-10
                
                # Predictive likelihood
                ll = stats.norm.logpdf(actual_err, loc=mu_simple, scale=sigma_simple)
                simple_cv_loglik += ll
                simple_predictions.append((actual_err, mu_simple, sigma_simple, point_idx))
            
        # CORRECTION-SPACE EVALUATION (joint likelihood per event)
        # Mean and covariance already computed above based on n_holdout_points
        ll_event = stats.multivariate_normal.logpdf(holdout_errors, mean=mean_vec, cov=cov)
        decomposed_cv_loglik += ll_event
        cv_used += 1
        
        # Store predictions for visualization (marginal per correction)
        for point_idx, (pc, actual_err) in enumerate(zip(holdout_postcals, holdout_errors), start=1):
            if n_holdout_points == 1:
                pred_mean = mu_total
                pred_std = sigma_total
            elif n_holdout_points == 2:
                pred_mean = mu_gain * pc
                pred_var = (sigma_gain * pc)**2 + sigma_residual**2
                pred_std = np.sqrt(pred_var)
            else:
                pred_mean = mu_offset + mu_gain * pc
                pred_var = sigma_offset**2 + (pc**2) * (sigma_gain**2) + sigma_noise**2
                pred_std = np.sqrt(pred_var)
            decomposed_predictions.append((actual_err, pred_mean, pred_std, point_idx))
    
    print(f"      Simple Model CV LogLik: {simple_cv_loglik:.2f}")
    print(f"      Decomposed Model CV LogLik: {decomposed_cv_loglik:.2f}")
    print(f"      CV folds used: {cv_used}, skipped: {cv_skipped}")
    
    # Detailed diagnostics
    if decomposed_predictions:
        pred_stds = [p[2] for p in decomposed_predictions]
        pred_errors = [abs(p[0] - p[1]) for p in decomposed_predictions]
        print(f"\n      Decomposed Model Diagnostics:")
        print(f"        Prediction std (σ): mean={np.mean(pred_stds):.6f}, min={np.min(pred_stds):.6f}, max={np.max(pred_stds):.6f}")
        print(f"        Prediction error: mean={np.mean(pred_errors):.6f}, median={np.median(pred_errors):.6f}")
        print(f"        Ratio (error/σ): {np.mean(pred_errors)/np.mean(pred_stds):.3f}")
        
        # Component variance statistics across folds
        if decomposed_component_variances:
            # Group by case
            cases = {}
            for d in decomposed_component_variances:
                case = d['case']
                if case not in cases:
                    cases[case] = []
                cases[case].append(d)
            
            print(f"        Component σ statistics by calibration type:")
            for case, fold_list in sorted(cases.items()):
                print(f"          {case}:")
                if case == '1-point':
                    sigmas = [d['sigma_total'] for d in fold_list if not np.isnan(d['sigma_total'])]
                    if sigmas:
                        print(f"            σ_total: mean={np.mean(sigmas):.6f}, range=[{np.min(sigmas):.6f}, {np.max(sigmas):.6f}]")
                elif case == '2-point':
                    sigma_gains = [d['sigma_gain'] for d in fold_list if not np.isnan(d['sigma_gain'])]
                    sigma_residuals = [d['sigma_residual'] for d in fold_list if not np.isnan(d['sigma_residual'])]
                    if sigma_gains:
                        print(f"            σ_gain: mean={np.mean(sigma_gains):.6f}, range=[{np.min(sigma_gains):.6f}, {np.max(sigma_gains):.6f}]")
                    if sigma_residuals:
                        print(f"            σ_residual: mean={np.mean(sigma_residuals):.6f}, range=[{np.min(sigma_residuals):.6f}, {np.max(sigma_residuals):.6f}]")
                elif case == '3+ point':
                    sigma_offsets = [d['sigma_offset'] for d in fold_list if not np.isnan(d['sigma_offset'])]
                    sigma_gains = [d['sigma_gain'] for d in fold_list if not np.isnan(d['sigma_gain'])]
                    sigma_noises = [d['sigma_noise'] for d in fold_list if not np.isnan(d['sigma_noise'])]
                    if sigma_offsets:
                        print(f"            σ_offset: mean={np.mean(sigma_offsets):.6f}, range=[{np.min(sigma_offsets):.6f}, {np.max(sigma_offsets):.6f}]")
                    if sigma_gains:
                        print(f"            σ_gain: mean={np.mean(sigma_gains):.6f}, range=[{np.min(sigma_gains):.6f}, {np.max(sigma_gains):.6f}]")
                    if sigma_noises:
                        print(f"            σ_noise: mean={np.mean(sigma_noises):.6f}, range=[{np.min(sigma_noises):.6f}, {np.max(sigma_noises):.6f}]")
    
    if simple_predictions:
        pred_stds_simple = [p[2] for p in simple_predictions]
        pred_errors_simple = [abs(p[0] - p[1]) for p in simple_predictions]
        print(f"\n      Simple Model Diagnostics:")
        print(f"        Prediction std (σ): mean={np.mean(pred_stds_simple):.6f}, min={np.min(pred_stds_simple):.6f}, max={np.max(pred_stds_simple):.6f}")
        print(f"        Prediction error: mean={np.mean(pred_errors_simple):.6f}, median={np.median(pred_errors_simple):.6f}")
        print(f"        Ratio (error/σ): {np.mean(pred_errors_simple)/np.mean(pred_stds_simple):.3f}")
    
    # Warn about positive log-likelihoods (now less likely with hierarchical evaluation)
    if simple_cv_loglik > 0:
        print(f"\n      ⚠ WARNING: Simple model has positive CV LogLik ({simple_cv_loglik:.2f})")
    if decomposed_cv_loglik > 0:
        print(f"\n      ⚠ WARNING: Decomposed model has positive CV LogLik ({decomposed_cv_loglik:.2f})")
    
    # ===== FIT FULL MODELS FOR PARAMETER COUNTING =====
    print(f"\n    - Fitting full models for parameter counting...")
    
    # Simple Model on all data
    print(f"    - Simple Model (independent distributions per point):")
    simple_loglik = 0
    simple_k = 0
    
    for point_idx, data in corrections_by_point.items():
        corrections = np.array([d[0] for d in data])
        n = len(corrections)
        
        if n < 2:
            print(f"      Point {point_idx}: SKIPPED (n={n}, insufficient data)")
            continue
        
        mu = np.mean(corrections)
        sigma = np.std(corrections, ddof=1)
        
        # Handle exactly zero variance
        if sigma == 0.0:
            sigma = 1e-10
        
        ll = np.sum(stats.norm.logpdf(corrections, loc=mu, scale=sigma))
        simple_loglik += ll
        simple_k += 2
        
        print(f"      Point {point_idx}: μ={mu:.4f}, σ={sigma:.4f}, LogLik={ll:.2f}")

    
    comparison['Simple_LogLik'] = simple_loglik
    comparison['Simple_K'] = simple_k
    comparison['Simple_AIC'] = 2 * simple_k - 2 * simple_loglik
    comparison['Simple_CV_LogLik'] = simple_cv_loglik
    
    print(f"      TOTAL: k={simple_k}, LogLik={simple_loglik:.2f}, AIC={comparison['Simple_AIC']:.2f}")
    print(f"      CV LogLik: {simple_cv_loglik:.2f}")

    
    # ===== DECOMPOSED MODEL =====
    print(f"    - Decomposed Model (hierarchical with point-count-dependent parameterization):")
    print(f"      1-point: σ_total | 2-point: σ_gain + σ_residual | 3+ point: σ_offset + σ_gain + σ_noise")
    
    # Separate events by calibration point count
    events_1pt = results_df[results_df['N_Points'] == 1]
    events_2pt = results_df[results_df['N_Points'] == 2]
    events_3plus = results_df[results_df['N_Points'] >= 3]
    
    print(f"      Event distribution: 1-pt={len(events_1pt)}, 2-pt={len(events_2pt)}, 3+pt={len(events_3plus)}")
    
    # Check if we have enough events
    total_events = len(results_df)
    if total_events < 10:
        print(f"    - WARNING: Not enough events to fit decomposed model (n={total_events}, require n≥10)")
        comparison['Recommendation'] = 'Simple'
        comparison['Reasons'] = f'Insufficient events for decomposed model (n={total_events} < 10)'
        return comparison
    
    # ===== 1-POINT EVENTS: Estimate σ_total =====
    if len(events_1pt) > 0:
        corrections_1pt = []
        for idx, row in events_1pt.iterrows():
            if 'Error_Values' in row and isinstance(row['Error_Values'], np.ndarray):
                corrections_1pt.extend(row['Error_Values'])
        
        if len(corrections_1pt) >= 2:
            mu_total = np.mean(corrections_1pt)
            sigma_total = np.std(corrections_1pt, ddof=1)
            if sigma_total == 0.0:
                sigma_total = 1e-10
            print(f"      1-point: μ_total={mu_total:.4f}, σ_total={sigma_total:.4f} (n={len(corrections_1pt)} corrections)")
        else:
            mu_total = np.nan
            sigma_total = np.nan
    else:
        mu_total = np.nan
        sigma_total = np.nan
    
    # ===== 2-POINT EVENTS: Estimate σ_gain + σ_residual =====
    if len(events_2pt) >= 2:
        gains_2pt = events_2pt['Gain'].dropna().values
        mu_gain_2pt = np.mean(gains_2pt)
        sigma_gain_2pt = np.std(gains_2pt, ddof=1)
        if sigma_gain_2pt == 0.0:
            sigma_gain_2pt = 1e-10
        
        # Compute residuals (offset + noise combined)
        residuals_2pt = []
        for idx, row in events_2pt.iterrows():
            if 'Error_Values' in row and 'PostCal_Values' in row and 'Gain' in row:
                errors = row['Error_Values']
                postcals = row['PostCal_Values']
                gain = row['Gain']
                if isinstance(errors, np.ndarray) and isinstance(postcals, np.ndarray):
                    residuals = errors - gain * postcals
                    residuals_2pt.extend(residuals)
        
        if len(residuals_2pt) >= 2:
            sigma_residual_2pt = np.std(residuals_2pt, ddof=1)
            if sigma_residual_2pt == 0.0:
                sigma_residual_2pt = 1e-10
        else:
            sigma_residual_2pt = 1e-10
        
        print(f"      2-point: μ_gain={mu_gain_2pt:.4f}, σ_gain={sigma_gain_2pt:.4f}, σ_residual={sigma_residual_2pt:.4f} (n={len(gains_2pt)} events, {len(residuals_2pt)} residuals)")
    else:
        mu_gain_2pt = np.nan
        sigma_gain_2pt = np.nan
        sigma_residual_2pt = np.nan
    
    # ===== 3+ POINT EVENTS: Estimate σ_offset + σ_gain + σ_noise =====
    if len(events_3plus) >= 2:
        offsets = events_3plus['Offset'].dropna().values
        gains = events_3plus['Gain'].dropna().values
        
        mu_offset = np.mean(offsets)
        sigma_offset = np.std(offsets, ddof=1)
        if sigma_offset == 0.0:
            sigma_offset = 1e-10
        
        mu_gain_3plus = np.mean(gains)
        sigma_gain_3plus = np.std(gains, ddof=1)
        if sigma_gain_3plus == 0.0:
            sigma_gain_3plus = 1e-10
        
        # Estimate noise from residuals
        residuals_3plus = []
        for idx, row in events_3plus.iterrows():
            if 'Residuals' in row and isinstance(row['Residuals'], np.ndarray):
                if len(row['Residuals']) > 0:
                    residuals_3plus.extend(row['Residuals'])
        
        if len(residuals_3plus) >= 2:
            sigma_noise = np.std(residuals_3plus, ddof=1)
            if sigma_noise == 0.0:
                sigma_noise = 1e-10
        else:
            sigma_noise = 1e-10
        
        print(f"      3+ point: μ_offset={mu_offset:.4f}, σ_offset={sigma_offset:.4f}, μ_gain={mu_gain_3plus:.4f}, σ_gain={sigma_gain_3plus:.4f}, σ_noise={sigma_noise:.4f}")
        print(f"               (n={len(offsets)} events, {len(residuals_3plus)} residuals)")
    else:
        mu_offset = np.nan
        sigma_offset = np.nan
        mu_gain_3plus = np.nan
        sigma_gain_3plus = np.nan
        sigma_noise = np.nan
    
    # Calculate log-likelihood in correction-space (appropriate covariance for each event)
    decomposed_loglik = 0
    decomposed_k = 0
    
    # 1-point events
    if not np.isnan(mu_total) and not np.isnan(sigma_total):
        for idx, row in events_1pt.iterrows():
            if 'Error_Values' not in row:
                continue
            errors = row['Error_Values']
            if not isinstance(errors, np.ndarray):
                continue
            
            mean_vec = np.array([mu_total])
            cov = np.array([[sigma_total**2 + 1e-12]])
            ll_event = stats.multivariate_normal.logpdf(errors, mean=mean_vec, cov=cov)
            decomposed_loglik += ll_event
        decomposed_k += 2  # μ_total, σ_total
    
    # 2-point events
    if not np.isnan(mu_gain_2pt) and not np.isnan(sigma_gain_2pt) and not np.isnan(sigma_residual_2pt):
        for idx, row in events_2pt.iterrows():
            if 'PostCal_Values' not in row or 'Error_Values' not in row:
                continue
            postcals = row['PostCal_Values']
            errors = row['Error_Values']
            if not isinstance(postcals, np.ndarray) or not isinstance(errors, np.ndarray):
                continue
            
            mean_vec = mu_gain_2pt * postcals
            pc_outer = np.outer(postcals, postcals)
            cov = (sigma_gain_2pt**2) * pc_outer + (sigma_residual_2pt**2) * np.eye(2)
            cov += np.eye(2) * 1e-12
            ll_event = stats.multivariate_normal.logpdf(errors, mean=mean_vec, cov=cov)
            decomposed_loglik += ll_event
        decomposed_k += 3  # μ_gain, σ_gain, σ_residual
    
    # 3+ point events
    if not np.isnan(mu_offset) and not np.isnan(sigma_offset) and not np.isnan(mu_gain_3plus) and not np.isnan(sigma_gain_3plus) and not np.isnan(sigma_noise):
        for idx, row in events_3plus.iterrows():
            if 'PostCal_Values' not in row or 'Error_Values' not in row:
                continue
            postcals = row['PostCal_Values']
            errors = row['Error_Values']
            if not isinstance(postcals, np.ndarray) or not isinstance(errors, np.ndarray):
                continue
            
            n_pts = len(postcals)
            mean_vec = mu_offset + mu_gain_3plus * postcals
            ones = np.ones((n_pts, n_pts))
            pc_outer = np.outer(postcals, postcals)
            cov = (sigma_offset**2) * ones + (sigma_gain_3plus**2) * pc_outer + (sigma_noise**2) * np.eye(n_pts)
            cov += np.eye(n_pts) * 1e-12
            ll_event = stats.multivariate_normal.logpdf(errors, mean=mean_vec, cov=cov)
            decomposed_loglik += ll_event
        decomposed_k += 5  # μ_offset, σ_offset, μ_gain, σ_gain, σ_noise
    
    comparison['Decomposed_LogLik'] = decomposed_loglik
    comparison['Decomposed_K'] = decomposed_k
    comparison['Decomposed_AIC'] = 2 * decomposed_k - 2 * decomposed_loglik
    comparison['Decomposed_CV_LogLik'] = decomposed_cv_loglik
    
    print(f"      TOTAL: k={decomposed_k}, LogLik={decomposed_loglik:.2f}, AIC={comparison['Decomposed_AIC']:.2f}")
    print(f"      CV LogLik: {decomposed_cv_loglik:.2f}")


    
    # ===== COMPARISON =====
    comparison['Delta_AIC'] = comparison['Decomposed_AIC'] - comparison['Simple_AIC']
    comparison['Delta_CV_LogLik'] = decomposed_cv_loglik - simple_cv_loglik
    
    # Primary comparison: Cross-validated log-likelihood (higher is better)
    # Secondary: AIC on full data (lower is better)
    if abs(comparison['Delta_CV_LogLik']) < 2:
        cv_conclusion = 'Equivalent'
    elif comparison['Delta_CV_LogLik'] > 2:
        cv_conclusion = 'Decomposed'
    else:
        cv_conclusion = 'Simple'
    
    # AIC interpretation: ΔAIC > 2 indicates substantial support for lower AIC model
    if abs(comparison['Delta_AIC']) < 2:
        aic_conclusion = 'Equivalent'
    elif comparison['Delta_AIC'] < -2:
        aic_conclusion = 'Decomposed'
    else:
        aic_conclusion = 'Simple'
    
    # Use CV as primary criterion
    comparison['Recommendation'] = cv_conclusion
    if cv_conclusion == 'Decomposed':
        comparison['Reasons'] = f'ΔCV_LogLik={comparison["Delta_CV_LogLik"]:.1f} (decomposed better by CV)'
    elif cv_conclusion == 'Simple':
        comparison['Reasons'] = f'ΔCV_LogLik={comparison["Delta_CV_LogLik"]:.1f} (simple better by CV)'
    else:
        comparison['Reasons'] = f'ΔCV_LogLik={comparison["Delta_CV_LogLik"]:.1f} (models equivalent by CV)'
    
    # Print comparison summary
    print(f"\n  → COMPARISON SUMMARY:")
    print(f"      ΔCV LogLik = {comparison['Delta_CV_LogLik']:.2f} (Decomposed - Simple)")
    print(f"      ΔAIC = {comparison['Delta_AIC']:.2f} (Decomposed - Simple)")
    print(f"      CV Recommendation: {cv_conclusion}")
    print(f"      AIC Recommendation: {aic_conclusion}")
    print(f"      Final: {comparison['Recommendation']} - {comparison['Reasons']}")
    
    # Store predictions for visualization
    comparison['Simple_Predictions'] = simple_predictions
    comparison['Decomposed_Predictions'] = decomposed_predictions
    
    return comparison


def create_model_comparison_visualization(comparison_df, output_dir=OUTPUT_DIR):
    """
    Visualize decomposed vs simple model comparison using AIC.
    
    Creates a figure showing:
    - AIC comparison (lower is better)
    - ΔAIC values (decomposed - simple)
    - Log-likelihoods
    - Model complexity (number of parameters)
    - Recommendation summary
    """
    fig = plt.figure(figsize=(20, 10), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    sensors = comparison_df['Sensor'].values
    n_sensors = len(sensors)
    x = np.arange(n_sensors)
    
    # Panel 1: AIC comparison (lower is better)
    ax1 = fig.add_subplot(gs[0, 0])
    simple_aic = comparison_df['Simple_AIC'].values
    decomp_aic = comparison_df['Decomposed_AIC'].values
    
    valid_idx = ~(np.isnan(simple_aic) | np.isnan(decomp_aic))
    if valid_idx.any():
        width = 0.35
        ax1.bar(x[valid_idx] - width/2, simple_aic[valid_idx], width,
                label='Simple', color='#1f77b4', alpha=0.7)
        ax1.bar(x[valid_idx] + width/2, decomp_aic[valid_idx], width,
                label='Decomposed', color='#ff7f0e', alpha=0.7)
        ax1.legend(fontsize=9, framealpha=0.9)
    else:
        ax1.text(n_sensors/2, 0.5, 'No valid data', ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
    
    ax1.set_ylabel('AIC (lower = better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sensors, rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Panel 2: ΔAIC (decomposed - simple)
    ax2 = fig.add_subplot(gs[0, 1])
    delta_aic = comparison_df['Delta_AIC'].values
    
    valid_idx = ~np.isnan(delta_aic)
    if valid_idx.any():
        colors = ['#2ca02c' if d < -2 else '#d62728' if d > 2 else '#ff7f0e' 
                  for d in delta_aic[valid_idx]]
        ax2.bar(x[valid_idx], delta_aic[valid_idx], color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.axhline(-2, color='green', linestyle='--', linewidth=1.2, alpha=0.5, label='Decomposed better')
        ax2.axhline(2, color='red', linestyle='--', linewidth=1.2, alpha=0.5, label='Simple better')
        ax2.legend(fontsize=9, framealpha=0.9)
    else:
        ax2.text(n_sensors/2, 0, 'No valid data', ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
    
    ax2.set_ylabel('ΔAIC\n(Decomposed − Simple)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sensors, rotation=45, ha='right', fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Panel 3: Number of parameters
    ax3 = fig.add_subplot(gs[0, 2])
    simple_k = comparison_df['Simple_K'].values
    decomp_k = comparison_df['Decomposed_K'].values
    
    valid_idx = ~(np.isnan(simple_k) | np.isnan(decomp_k))
    if valid_idx.any():
        width = 0.35
        ax3.bar(x[valid_idx] - width/2, simple_k[valid_idx], width,
                label='Simple', color='#1f77b4', alpha=0.7)
        ax3.bar(x[valid_idx] + width/2, decomp_k[valid_idx], width,
                label='Decomposed', color='#ff7f0e', alpha=0.7)
        ax3.legend(fontsize=9, framealpha=0.9)
    else:
        ax3.text(n_sensors/2, 2, 'No valid data', ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
    
    ax3.set_ylabel('Model Complexity\n(# parameters)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sensors, rotation=45, ha='right', fontsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.grid(True, alpha=0.3, linewidth=0.8)
    ax3.set_axisbelow(True)
    
    # Panel 4: Log-likelihood (higher is better)
    ax4 = fig.add_subplot(gs[1, 0])
    simple_ll = comparison_df['Simple_LogLik'].values
    decomp_ll = comparison_df['Decomposed_LogLik'].values
    
    valid_idx = ~(np.isnan(simple_ll) | np.isnan(decomp_ll))
    if valid_idx.any():
        width = 0.35
        ax4.bar(x[valid_idx] - width/2, simple_ll[valid_idx], width,
                label='Simple', color='#1f77b4', alpha=0.7)
        ax4.bar(x[valid_idx] + width/2, decomp_ll[valid_idx], width,
                label='Decomposed', color='#ff7f0e', alpha=0.7)
        ax4.legend(fontsize=9, framealpha=0.9)
    else:
        ax4.text(n_sensors/2, 0, 'No valid data', ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')
    
    ax4.set_ylabel('Log-Likelihood\n(higher = better fit)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sensors, rotation=45, ha='right', fontsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    ax4.grid(True, alpha=0.3, linewidth=0.8)
    ax4.set_axisbelow(True)
    
    # Panel 5: Recommendation summary (text table)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create table data
    table_data = []
    for i, row in comparison_df.iterrows():
        rec = row['Recommendation']
        reasons = row['Reasons']
        n_corr = row.get('N_Corrections', '?')
        # Truncate reasons if too long
        if len(str(reasons)) > 50:
            reasons = str(reasons)[:47] + "..."
        table_data.append([row['Sensor'], f"n={n_corr}", rec, str(reasons)])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Sensor', 'Data', 'Recommendation', 'Reason'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.2, 0.1, 0.2, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code recommendations
    for i in range(1, len(table_data) + 1):
        rec_cell = table[(i, 2)]
        rec_val = table_data[i-1][2]
        if rec_val == 'Simple':
            rec_cell.set_facecolor('#FFB6C1')  # Light red
        elif rec_val == 'Decomposed':
            rec_cell.set_facecolor('#90EE90')  # Light green
        else:
            rec_cell.set_facecolor('#FFFFE0')  # Light yellow (equivalent)
    
    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    # Remove titles before saving
    clear_figure_titles(fig)
    
    output_path = output_dir / 'aggregate' / 'model_comparison_summary.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: model_comparison_summary.png")


def create_prediction_error_visualization(comparison_result, sensor_name, output_dir=OUTPUT_DIR):
    """
    Create visualization comparing prediction errors from both models.
    
    Shows:
    - Histogram of prediction errors (actual - predicted) for both models
    - Scatter plot of predicted vs actual values
    - Residual plots
    """
    simple_preds = comparison_result.get('Simple_Predictions', [])
    decomposed_preds = comparison_result.get('Decomposed_Predictions', [])
    
    if not simple_preds or not decomposed_preds:
        return
    
    # Extract data
    simple_actual = np.array([p[0] for p in simple_preds])
    simple_pred = np.array([p[1] for p in simple_preds])
    simple_std = np.array([p[2] for p in simple_preds])
    
    decomposed_actual = np.array([p[0] for p in decomposed_preds])
    decomposed_pred = np.array([p[1] for p in decomposed_preds])
    decomposed_std = np.array([p[2] for p in decomposed_preds])
    
    # Calculate errors
    simple_errors = simple_actual - simple_pred
    decomposed_errors = decomposed_actual - decomposed_pred
    
    # Create figure
    fig = plt.figure(figsize=(20, 12), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)
    
    fig.suptitle(f'{sensor_name} - Cross-Validated Prediction Accuracy', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # ===== ROW 1: PREDICTION ERROR HISTOGRAMS =====
    
    # Panel 1: Simple Model Errors
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(simple_errors, bins=max(10, len(simple_errors)//5), 
             alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.8)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(np.mean(simple_errors), color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Prediction Error (Actual - Predicted)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title(f'Simple Model Errors\\nMean={np.mean(simple_errors):.4f}, Std={np.std(simple_errors):.4f}',
                  fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Decomposed Model Errors
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(decomposed_errors, bins=max(10, len(decomposed_errors)//5),
             alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(np.mean(decomposed_errors), color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Prediction Error (Actual - Predicted)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title(f'Decomposed Model Errors\\nMean={np.mean(decomposed_errors):.4f}, Std={np.std(decomposed_errors):.4f}',
                  fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Error Comparison (Side by Side)
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(min(simple_errors.min(), decomposed_errors.min()),
                       max(simple_errors.max(), decomposed_errors.max()), 20)
    ax3.hist(simple_errors, bins=bins, alpha=0.5, color='#1f77b4', 
             label='Simple', edgecolor='black', linewidth=0.8)
    ax3.hist(decomposed_errors, bins=bins, alpha=0.5, color='#2ca02c',
             label='Decomposed', edgecolor='black', linewidth=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Prediction Error', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Error Distribution Comparison', fontweight='bold', fontsize=11)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # ===== ROW 2: PREDICTED VS ACTUAL =====
    
    # Panel 4: Simple Model - Predicted vs Actual
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(simple_actual, simple_pred, alpha=0.6, s=40, color='#1f77b4', edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    all_vals = np.concatenate([simple_actual, simple_pred])
    lims = [all_vals.min(), all_vals.max()]
    ax4.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # Calculate R²
    r2_simple = 1 - np.sum(simple_errors**2) / np.sum((simple_actual - np.mean(simple_actual))**2)
    rmse_simple = np.sqrt(np.mean(simple_errors**2))
    
    ax4.set_xlabel('Actual Correction', fontweight='bold')
    ax4.set_ylabel('Predicted Correction', fontweight='bold')
    ax4.set_title(f'Simple Model\\nR²={r2_simple:.3f}, RMSE={rmse_simple:.4f}',
                  fontweight='bold', fontsize=11)
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    # Panel 5: Decomposed Model - Predicted vs Actual
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(decomposed_actual, decomposed_pred, alpha=0.6, s=40, color='#2ca02c', 
                edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    all_vals = np.concatenate([decomposed_actual, decomposed_pred])
    lims = [all_vals.min(), all_vals.max()]
    ax5.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # Calculate R²
    r2_decomp = 1 - np.sum(decomposed_errors**2) / np.sum((decomposed_actual - np.mean(decomposed_actual))**2)
    rmse_decomp = np.sqrt(np.mean(decomposed_errors**2))
    
    ax5.set_xlabel('Actual Correction', fontweight='bold')
    ax5.set_ylabel('Predicted Correction', fontweight='bold')
    ax5.set_title(f'Decomposed Model\\nR²={r2_decomp:.3f}, RMSE={rmse_decomp:.4f}',
                  fontweight='bold', fontsize=11)
    ax5.legend(loc='upper left', framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    # Panel 6: Model Comparison Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create comparison text
    delta_cv = comparison_result.get('Delta_CV_LogLik', np.nan)
    delta_aic = comparison_result.get('Delta_AIC', np.nan)
    recommendation = comparison_result.get('Recommendation', 'Unknown')
    
    summary_text = f"""MODEL COMPARISON SUMMARY
    
Cross-Validation:
  Simple CV LogLik:     {comparison_result.get('Simple_CV_LogLik', np.nan):.2f}
  Decomposed CV LogLik: {comparison_result.get('Decomposed_CV_LogLik', np.nan):.2f}
  ΔCV LogLik:           {delta_cv:.2f}
  
Predictive Accuracy:
  Simple RMSE:          {rmse_simple:.4f}
  Decomposed RMSE:      {rmse_decomp:.4f}
  Improvement:          {((rmse_simple - rmse_decomp)/rmse_simple * 100):.1f}%
  
  Simple R²:            {r2_simple:.3f}
  Decomposed R²:        {r2_decomp:.3f}
  
Information Criteria:
  Simple AIC:           {comparison_result.get('Simple_AIC', np.nan):.2f}
  Decomposed AIC:       {comparison_result.get('Decomposed_AIC', np.nan):.2f}
  ΔAIC:                 {delta_aic:.2f}
  
Sample Size:
  N Events:             {comparison_result.get('N_Events', 0)}
  N Corrections:        {comparison_result.get('N_Corrections', 0)}
  
RECOMMENDATION: {recommendation}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Remove titles before saving
    clear_figure_titles(fig)
    
    output_path = output_dir / f'prediction_errors_{sensor_name}.png'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  - Saved prediction_errors_{sensor_name}.png")


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
    all_raw_data = {}  # Store raw_df for model comparison
    
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
                all_raw_data[sensor_name] = raw_df  # Store for model comparison
                
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
            
            # ===== KRUSKAL-WALLIS TEST ACROSS CORRECTIONS =====
            print(f"  - Testing if corrections across points have same distribution...")
            kw_result = test_corrections_kruskal_wallis(raw_df, sensor_name)
            if kw_result['Test_Status'] == 'Completed':
                kw_result['Sensor'] = sensor_name
                kw_result['Calibration_Point'] = 'Across All Points'
                all_separate_stats.append(kw_result)
                print(f"    - Kruskal-Wallis: H={kw_result['Kruskal_Wallis_H']:.4f}, p={kw_result['Kruskal_Wallis_p']:.4f}, " +
                      f"Significant={kw_result['Kruskal_Wallis_Significant']}")
                print(f"    - ANOVA: F={kw_result['ANOVA_F']:.4f}, p={kw_result['ANOVA_p']:.4f}, " +
                      f"Significant={kw_result['ANOVA_Significant']}, η²={kw_result['Eta_Squared']:.4f}")
                print(f"    - Levene: F={kw_result['Levene_F']:.4f}, p={kw_result['Levene_p']:.4f}, " +
                      f"Equal Variances={not kw_result['Levene_Significant']}")
            else:
                print(f"    - {kw_result['Test_Status']} (need ≥2 points with ≥5 observations each)")
            
            # Create visualization of corrections comparison
            create_corrections_comparison_visualization(raw_df, sensor_name, output_dir=sensor_output_dir)
            
            # ===== SAVE UNCERTAINTY SUMMARY FOR MONTE CARLO SAMPLING =====
            n_correction_cols = len([col for col in raw_df.columns if col.startswith('Correction')])
            if len(og_df) > 0:
                save_sensor_uncertainty_summary(og_df, sensor_name, n_correction_cols, output_dir=OUTPUT_DIR)
            
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
    
    # ===== MODEL COMPARISON =====
    if all_offset_gain_results and all_raw_data:
        print("\n" + "="*80)
        print("MODEL COMPARISON: Decomposed vs Simple")
        print("="*80)
        comparison_results = []
        
        for sensor_name in sorted(all_raw_data.keys()):
            print(f"Comparing models for: {sensor_name}")
            # Get results and raw data for this sensor
            sensor_results = og_combined[og_combined['Sensor'] == sensor_name]
            raw_df = all_raw_data[sensor_name]
            
            # Create sensor-specific output directory
            sensor_output_dir = OUTPUT_DIR / sensor_name
            sensor_output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison = compare_decomposed_vs_simple_model(sensor_results, raw_df, sensor_name)
            comparison_results.append(comparison)
            
            # Create prediction error visualization for this sensor in sensor-specific directory
            create_prediction_error_visualization(comparison, sensor_name, sensor_output_dir)
            
            rec = comparison['Recommendation']
            print(f"  -> Recommendation: {rec}")
            print(f"     {comparison['Reasons']}")
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison results (remove prediction lists before saving)
        comparison_save = comparison_df.drop(columns=['Simple_Predictions', 'Decomposed_Predictions'], errors='ignore')
        comparison_path = aggregate_dir / 'model_comparison.csv'
        comparison_save.to_csv(comparison_path, index=False)
        print(f"\n  - Saved model_comparison.csv ({len(comparison_save)} sensors)")
        
        # Create visualization
        create_model_comparison_visualization(comparison_save, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Aggregate results saved to: {aggregate_dir}")


if __name__ == "__main__":
    main()

