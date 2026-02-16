"""
Verify Monte Carlo perturbations by comparing distributions of perturbed vs unperturbed values.
Throwaway script for validation purposes.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def find_valid(df, targets, predictors, span, nan_tol):
    """Find valid segment ending indices (copied from d_Resample.py)."""
    valid_indices = []
    for i in range(len(df)):
        # Check targets in current row (use iloc for positional indexing)
        if df.iloc[i][targets].isna().any():
            continue
        
        # Define window for previous rows
        start = max(0, i - span)
        window = df.iloc[start:i]
        
        if window.empty:
            continue
        
        # Count non-NaN predictor values in the window
        total_values = len(window) * len(predictors)
        non_nan_values = window[predictors].notna().sum().sum()
        
        # Check if proportion meets threshold
        if total_values > 0 and 1 - (non_nan_values / total_values) <= nan_tol:
            valid_indices.append(i)
    
    return valid_indices

def compare_perturbation_magnitudes(sensor_name, segment_num, original_df, replicate_dfs, column_name, uncertainty_params, norm_params):
    """
    Compare perturbation magnitude by analyzing per-row variation across replicates.
    
    For each row in the original segment:
    - Get original value
    - Get 10 perturbed versions from replicates (matched by row position, not index)
    - Compute std of those 10 values (perturbation magnitude)
    - Track how many rows had sufficient data
    
    Returns: dict with comparison statistics
    
    Note: Replicates are matched by iloc position since they don't have DatetimeIndex
    
    Expected perturbation accounts for:
    - Offset variance (drawn per replicate): Var(offset)
    
    Note: Gain and noise perturbations have been removed from the MC process,
    so only offset variance is included in expected perturbation.
    """
    if column_name not in original_df.columns:
        return None
    
    # Get original values
    orig_values = original_df[column_name].copy()
    
    # Find rows with valid original values (using position/iloc)
    valid_positions = []
    for pos in range(len(original_df)):
        if pd.notna(orig_values.iloc[pos]):
            valid_positions.append(pos)
    
    if len(valid_positions) == 0:
        return None
    
    # For each valid row position, collect its 10 perturbed versions
    per_row_perturbation_stds = []
    rows_with_data = 0
    rows_with_missing_replicates = 0
    
    for pos in valid_positions:
        # Get original value at this position
        orig_val = orig_values.iloc[pos]
        
        # Get perturbed versions from replicates (matched by position)
        perturbed_vals = []
        for replicate_df in replicate_dfs:
            if pos < len(replicate_df) and column_name in replicate_df.columns:
                pert_val = replicate_df.iloc[pos][column_name]
                if pd.notna(pert_val):
                    perturbed_vals.append(float(pert_val))
        
        # Only use if we have at least 3 replicates (min for meaningful statistics)
        if len(perturbed_vals) >= 3:
            # Compute std of this row's perturbations
            row_perturbation_std = np.std(perturbed_vals)
            per_row_perturbation_stds.append(row_perturbation_std)
            rows_with_data += 1
        else:
            rows_with_missing_replicates += 1
    
    if len(per_row_perturbation_stds) == 0:
        return None
    
    # Aggregate: average std across all rows
    avg_perturbation_std = np.mean(per_row_perturbation_stds)
    
    stats_dict = {
        'sensor': sensor_name,
        'segment_num': segment_num,
        'rows_with_full_data': rows_with_data,
        'rows_with_missing_replicates': rows_with_missing_replicates,
        'empirical_perturbation_std_mean': avg_perturbation_std,
        'empirical_perturbation_std_median': np.median(per_row_perturbation_stds),
        'empirical_perturbation_std_min': np.min(per_row_perturbation_stds),
        'empirical_perturbation_std_max': np.max(per_row_perturbation_stds),
        'per_row_stds_array': per_row_perturbation_stds,
    }
    
    # Extract theoretical parameters (offset-only model)
    n_cal_points = uncertainty_params.get('N_Calibration_Points', 0)
    offset_std = uncertainty_params.get('Offset_Std', 0)
    
    stats_dict['n_calibration_points'] = n_cal_points
    stats_dict['offset_std'] = offset_std
    
    # Build expected perturbation for offset-only model
    # Per-replicate variance sources:
    # - Offset drawn once per replicate from Correction1 distribution: contributes Var(offset) = offset_std^2
    # Note: Gain and noise have been removed from the perturbation process
    
    expected_components = []
    expected_variance = 0.0
    
    # Offset (based on Correction1 distribution)
    if offset_std > 0:
        expected_components.append(f"offset({offset_std:.4f})")
        expected_variance += offset_std ** 2
    
    expected_std = np.sqrt(expected_variance) if expected_variance > 0 else 0
    stats_dict['components'] = ', '.join(expected_components) if expected_components else 'none'
    stats_dict['expected_perturbation_std'] = expected_std
    
    if expected_std > 0:
        stats_dict['std_ratio'] = stats_dict['empirical_perturbation_std_mean'] / expected_std
    else:
        stats_dict['std_ratio'] = np.nan
    
    return stats_dict

def load_normalization_params(normalization_json_path):
    """Load min-max normalization parameters."""
    try:
        with open(normalization_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load normalization params from {normalization_json_path}: {e}")
        return {}

def normalize_uncertainty_params(params, col_name, norm_params):
    """
    Transform uncertainty parameters from raw scale to normalized scale.
    
    For min-max normalization with range = max - min:
    - If offset ~ N(mu, sigma) in raw scale,
      then normalized_offset ~ N(mu/range, sigma/range) in normalized scale
    - Gain is multiplicative and remains unchanged
    """
    if col_name not in norm_params:
        return params  # No normalization, return as-is
    
    norm_spec = norm_params[col_name]
    v_min = norm_spec.get('min', 0)
    v_max = norm_spec.get('max', 1)
    v_range = v_max - v_min
    
    if v_range == 0:
        return params
    
    # Create a copy so we don't modify original
    params_normalized = params.copy()
    
    # Scale additive components (offset, noise) by range
    if 'Offset_Mean' in params_normalized:
        params_normalized['Offset_Mean'] = params_normalized.get('Offset_Mean', 0) / v_range
    if 'Offset_Std' in params_normalized:
        params_normalized['Offset_Std'] = params_normalized.get('Offset_Std', 0) / v_range
    if 'Noise_Std_Mean' in params_normalized:
        params_normalized['Noise_Std_Mean'] = params_normalized.get('Noise_Std_Mean', 0) / v_range
    if 'Noise_Std_Std' in params_normalized:
        params_normalized['Noise_Std_Std'] = params_normalized.get('Noise_Std_Std', 0) / v_range
    
    # Gain is multiplicative, so it's unchanged
    # Gain affects (value * (1 + gain)), so normalized gain is also (1+gain)
    
    return params_normalized

def plot_before_after_distributions(sensor_name, original_values, perturbed_values, uncertainty_params, output_dir):
    """
    Create comprehensive before/after comparison plots.
    Shows original distribution vs perturbed distribution to detect over-perturbation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'{sensor_name}: Original vs Perturbed Distributions', fontsize=14, fontweight='bold')
    
    # Row 1: Histograms
    ax = axes[0, 0]
    ax.hist(original_values, bins=50, alpha=0.6, color='blue', label='Original', edgecolor='black')
    ax.axvline(np.mean(original_values), color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Original Distribution (n={len(original_values)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(perturbed_values, bins=50, alpha=0.6, color='red', label='Perturbed', edgecolor='black')
    ax.axvline(np.mean(perturbed_values), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Perturbed Distribution (n={len(perturbed_values)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Q-Q plots
    ax = axes[1, 0]
    stats.probplot(original_values, dist="norm", plot=ax)
    ax.set_title('Original: Q-Q Plot vs Normal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    stats.probplot(perturbed_values, dist="norm", plot=ax)
    ax.set_title('Perturbed: Q-Q Plot vs Normal')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Overlay and difference
    ax = axes[2, 0]
    ax.hist(original_values, bins=50, alpha=0.5, color='blue', label='Original', density=True, edgecolor='black')
    ax.hist(perturbed_values, bins=50, alpha=0.5, color='red', label='Perturbed', density=True, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Overlay Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Perturbation magnitude (difference) histogram
    ax = axes[2, 1]
    # Approximate perturbation: perturbed values relative to original distribution center
    # Note: This is not per-value matched, but shows aggregate shift pattern
    perturbation_delta = perturbed_values - np.mean(original_values)
    ax.hist(perturbation_delta, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    offset_mean = uncertainty_params.get('Offset_Mean', 0)
    ax.axvline(offset_mean, color='purple', linestyle='--', linewidth=2, label=f'Expected offset: {offset_mean:.4f}')
    ax.set_xlabel('Value - Original Mean (approximate perturbation)')
    ax.set_ylabel('Frequency')
    ax.set_title('Perturbation Shift Distribution (Approximate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"{sensor_name}_before_after_comparison.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    return {
        'original_mean': np.mean(original_values),
        'original_std': np.std(original_values),
        'perturbed_mean': np.mean(perturbed_values),
        'perturbed_std': np.std(perturbed_values),
        'mean_shift': np.mean(perturbed_values) - np.mean(original_values),
        'std_change': np.std(perturbed_values) - np.std(original_values),
    }

def plot_distribution_diagnostics(sensor_name, original_values, perturbed_values, params, output_dir):
    """
    Generate diagnostic statistics about distribution changes.
    """
    diagnostics = {
        'sensor': sensor_name,
    }
    
    original_mean = np.mean(original_values)
    original_std = np.std(original_values)
    perturbed_mean = np.mean(perturbed_values)
    perturbed_std = np.std(perturbed_values)
    
    diagnostics['original_mean'] = original_mean
    diagnostics['original_std'] = original_std
    diagnostics['perturbed_mean'] = perturbed_mean
    diagnostics['perturbed_std'] = perturbed_std
    diagnostics['mean_shift'] = perturbed_mean - original_mean
    diagnostics['std_change_percent'] = 100 * (perturbed_std - original_std) / original_std if original_std > 0 else 0
    
    # Expected changes from uncertainty parameters
    expected_offset = params.get('Offset_Mean', 0)
    expected_offset_std = params.get('Offset_Std', 0)
    expected_noise_std = params.get('Noise_Std_Mean', 0)
    
    diagnostics['expected_offset_mean'] = expected_offset
    diagnostics['expected_offset_std'] = expected_offset_std
    diagnostics['expected_noise_std'] = expected_noise_std
    diagnostics['expected_std_increase'] = expected_noise_std  # Assuming original doesn't change much
    
    # Shapiro-Wilk test for normality
    if len(original_values) > 5000:
        # Sample for large datasets
        orig_sample = np.random.choice(original_values, 5000, replace=False)
        pert_sample = np.random.choice(perturbed_values, 5000, replace=False)
    else:
        orig_sample = original_values
        pert_sample = perturbed_values
    
    try:
        orig_shapiro_p = stats.shapiro(orig_sample)[1]
        pert_shapiro_p = stats.shapiro(pert_sample)[1]
        diagnostics['original_shapiro_p'] = orig_shapiro_p
        diagnostics['perturbed_shapiro_p'] = pert_shapiro_p
    except Exception as e:
        print(f"    Warning: Shapiro-Wilk test failed for {sensor_name}: {e}")
        diagnostics['original_shapiro_p'] = np.nan
        diagnostics['perturbed_shapiro_p'] = np.nan
    
    return diagnostics

def main():
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    # Resampling parameters (should match d_Resample.py)
    SEGMENT_LENGTH = 168  # hours (7 days)
    NAN_TOLERANCE = 0.50  # max fraction of NaN values allowed
    
    # Monte Carlo parameters
    N_MC_REPLICATES = 10  # number of MC replicates per segment
    MIN_REPLICATES_FOR_STATS = 3  # minimum replicates needed for meaningful statistics
    MAX_SEGMENTS = None  # maximum segments to process (None = all available)
    
    # Validation thresholds
    RATIO_OK_MIN = 0.8
    RATIO_OK_MAX = 1.2
    RATIO_WARN_MIN = 0.6
    RATIO_WARN_MAX = 1.5
    
    # Logging
    LOG_FREQUENCY = 10  # log every N segments
    LOG_FIRST_N = 5  # always log first N segments
    
    # Target variable
    TARGET_COLUMNS = ['pH']
    
    # Paths - resolve relative to this script's location
    script_dir = Path(__file__).parent
    consolidated_csv = script_dir / "data" / "output" / "regression" / "Consolidated_sparse.csv"
    samples_dir = script_dir / "data" / "output" / "regression" / "MC_pH" / "samples"
    normalization_json = script_dir / "data" / "input" / "normalization.json"
    output_dir = script_dir / "data" / "output" / "regression" / "MC_pH" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    # ============================================================================
    
    # Load normalization parameters
    print("Loading normalization parameters...")
    norm_params = load_normalization_params(str(normalization_json))
    print(f"  OK: Loaded normalization for {len(norm_params)} sensors\n")
    
    # Load original data
    print("Loading sensor uncertainty parameters from c2_uncertainty.py outputs...")
    corrections_dir = Path(__file__).parent / "data" / "output" / "calibration" / "summaries"
    
    sensor_mappings = {
        'Sp Cond (microS_cm)': 'Pfl - Sp Cond (microS_cm)',
        'pH': 'Pfl - pH',
        'DO (% Sat)': 'Pfl - DO (% Sat)',
        'Turbidity (FNU)': 'Pfl - Turbidity (FNU)',
        'fDOM (RFU)': 'Pfl - fDOM (RFU)',
        'fDOM (QSU)': 'Pfl - fDOM (QSU)',
    }
    
    # Load uncertainty parameters
    uncertainty_params = {}
    print("\nLoading uncertainty summaries:")
    for sensor_key, col_name in sensor_mappings.items():
        summary_path = corrections_dir / sensor_key / f'{sensor_key}_uncertainty_summary.csv'
        if summary_path.exists():
            params = pd.read_csv(summary_path).iloc[0].to_dict()
            # Transform to normalized scale for comparison with normalized data
            params_normalized = normalize_uncertainty_params(params, col_name, norm_params)
            uncertainty_params[sensor_key] = params_normalized
            print(f"  OK: Loaded {sensor_key} (transformed to normalized scale)")
        else:
            print(f"  WARNING: Uncertainty summary not found: {summary_path}")
    
    print()
    
    # Load consolidated CSV and find valid segments using same logic as d_Resample.py
    print("Loading original data and finding valid segments...")
    try:
        original_data = pd.read_csv(str(consolidated_csv), index_col=0, parse_dates=True)
    except Exception as e:
        print(f"ERROR: Could not load consolidated CSV: {e}")
        return
    
    # Normalize the original data to match what d_Resample.py does
    import sys
    sys.path.insert(0, str(script_dir / 'src'))
    try:
        from utils.preprocessing import normalize_columns
    except ImportError as e:
        print(f"ERROR: Could not import normalize_columns from utils.preprocessing: {e}")
        print("Make sure the src/utils directory is properly set up.")
        return
    
    predictor_cols = list(sensor_mappings.values())
    
    print("Normalizing original data...")
    original_data_normalized = normalize_columns(
        original_data.copy(), 
        predictor_cols, 
        param_file=None, 
        min_val=0, 
        max_val=1, 
        save=False, 
        directory=None
    )
    
    # Find valid segments using same parameters as d_Resample.py
    print(f"Finding valid segments (length={SEGMENT_LENGTH}, nan_tol={NAN_TOLERANCE})...")
    valid_indices = find_valid(original_data_normalized, TARGET_COLUMNS, predictor_cols, SEGMENT_LENGTH, NAN_TOLERANCE)
    print(f"Found {len(valid_indices)} valid segment indices")
    
    if len(valid_indices) == 0:
        print("ERROR: No valid segments found!")
        return
    
    # Determine how many segments to process
    if MAX_SEGMENTS is not None and MAX_SEGMENTS < len(valid_indices):
        segments_to_process = valid_indices[:MAX_SEGMENTS]
        print(f"Processing first {MAX_SEGMENTS} of {len(valid_indices)} valid segments (MAX_SEGMENTS limit)\n")
    else:
        segments_to_process = valid_indices
        print(f"Processing ALL {len(valid_indices)} valid segments\n")
    
    # Per-sensor aggregation: {sensor_key: list of results, one per segment}
    sensor_segment_results = {sensor_key: [] for sensor_key in sensor_mappings.keys()}
    
    # Collect all values for distribution plots
    sensor_original_values = {sensor_key: [] for sensor_key in sensor_mappings.keys()}
    sensor_perturbed_values = {sensor_key: [] for sensor_key in sensor_mappings.keys()}
    
    segments_checked = 0
    segments_skipped = 0
    segments_missing_files = 0
    
    print("Analyzing per-row perturbation magnitudes:\n")
    print(f"Validation: Checking that segment files match expected naming pattern...")
    print(f"  Expected: segment_NNNN_mc_KKK.csv where NNNN = segment number, KKK = replicate 001-{N_MC_REPLICATES:03d}\n")
    
    for i, idx in enumerate(segments_to_process):
        segment_num = i + 1  # Segment numbering: i-th valid index → segment_{i+1}
        
        # Extract segment using same logic as d_Resample.py
        start = max(0, idx - SEGMENT_LENGTH)
        end = idx + 1
        original_segment = original_data_normalized.iloc[start:end]
        
        if original_segment.empty:
            segments_skipped += 1
            continue
        
        # Load all MC replicate DataFrames for this segment
        replicate_dfs = []
        missing_files = []
        for k in range(1, N_MC_REPLICATES + 1):
            filename = samples_dir / f"segment_{segment_num:04d}_mc_{k:03d}.csv"
            if filename.exists():
                try:
                    rep_df = pd.read_csv(str(filename))
                    replicate_dfs.append(rep_df)
                except Exception as e:
                    print(f"  WARNING: Could not read {filename.name}: {e}")
                    missing_files.append(k)
            else:
                missing_files.append(k)
        
        if len(replicate_dfs) < N_MC_REPLICATES:
            if segments_missing_files == 0:  # Log first instance
                print(f"  WARNING: Segment {segment_num} missing replicates: {missing_files}")
            segments_missing_files += 1
            segments_skipped += 1
            continue
        
        segments_checked += 1
        if segments_checked % LOG_FREQUENCY == 0 or segments_checked <= LOG_FIRST_N:
            print(f"Segment {segment_num} (idx {idx}, rows {start}-{end}): Analyzing {len(replicate_dfs)} replicates")
        
        # For each sensor, collect values and compare per-row perturbations
        for sensor_key, col_name in sensor_mappings.items():
            if sensor_key not in uncertainty_params:
                continue
            
            # Collect values from valid positions (original is already normalized)
            if col_name not in original_segment.columns:
                continue
            
            # Get original values (already normalized) and collect them
            orig_vals = original_segment[col_name].dropna().values
            if len(orig_vals) > 0:
                sensor_original_values[sensor_key].extend(orig_vals)
            
            # Collect perturbed values from all replicates
            for rep_df in replicate_dfs:
                if col_name in rep_df.columns:
                    pert_vals = rep_df[col_name].dropna().values
                    sensor_perturbed_values[sensor_key].extend(pert_vals)
            
            result = compare_perturbation_magnitudes(
                sensor_key, 
                segment_num,
                original_segment, 
                replicate_dfs, 
                col_name,
                uncertainty_params[sensor_key],
                norm_params
            )
            
            if result is not None:
                sensor_segment_results[sensor_key].append(result)
    
    print(f"\n{'='*60}")
    print(f"SEGMENT PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Valid segment indices found: {len(valid_indices)}")
    print(f"Segments attempted: {len(segments_to_process)}")
    print(f"Segments successfully processed: {segments_checked}")
    print(f"Segments skipped: {segments_skipped}")
    print(f"  - Due to missing MC files: {segments_missing_files}")
    print(f"  - Due to empty data: {segments_skipped - segments_missing_files}")
    print(f"Success rate: {100*segments_checked/len(segments_to_process):.1f}%\n")
    
    if segments_missing_files > 0:
        print(f"WARNING: {segments_missing_files} segments were skipped due to missing MC replicate files.")
        print(f"Expected file pattern: segment_NNNN_mc_KKK.csv with KKK from 001 to {N_MC_REPLICATES:03d}")
        print(f"Check that d_Resample.py generated all expected files in: {samples_dir}")
        print(f"\nTo verify file availability, check: {samples_dir}")
        print(f"Expected {len(segments_to_process) * N_MC_REPLICATES} total files, found {segments_checked * N_MC_REPLICATES}\n")
    
    print("Aggregating results across segments:\n")
    
    # Aggregate results per sensor across all segments
    results = []
    for sensor_key, col_name in sensor_mappings.items():
        segment_results = sensor_segment_results[sensor_key]
        
        if not segment_results:
            print(f"  X {sensor_key}: No valid data across any segment")
            continue
        
        print(f"  {sensor_key}: {len(segment_results)} segments with data")
        
        # Average empirical perturbations across segments
        empirical_stds = [r['empirical_perturbation_std_mean'] for r in segment_results]
        rows_with_data_total = sum(r['rows_with_full_data'] for r in segment_results)
        
        params = uncertainty_params[sensor_key]
        n_cal = params.get('N_Calibration_Points', 0)
        components = segment_results[0].get('components', 'none')
        expected_std = segment_results[0].get('expected_perturbation_std', 0)
        
        print(f"    Calibration points: {n_cal}")
        print(f"    Components: {components}")
        print(f"    Total rows analyzed: {rows_with_data_total}")
        print(f"    Empirical perturbation std (avg): {np.mean(empirical_stds):.6f}")
        print(f"    Expected perturbation std: {expected_std:.6f}")
        
        if expected_std > 0:
            ratio = np.mean(empirical_stds) / expected_std
            status = "OK" if RATIO_OK_MIN <= ratio <= RATIO_OK_MAX else "WARN" if RATIO_WARN_MIN <= ratio <= RATIO_WARN_MAX else "FAIL"
            print(f"    Match ratio: {ratio:.4f} {status}")
        else:
            ratio = np.nan
            print(f"    Match ratio: N/A (no offset component expected)")
        
        # Create aggregate result
        agg_result = {
            'sensor': sensor_key,
            'n_segments': len(segment_results),
            'n_calibration_points': n_cal,
            'components': components,
            'empirical_perturbation_std_mean': np.mean(empirical_stds),
            'empirical_perturbation_std_median': np.median(empirical_stds),
            'empirical_perturbation_std_min': np.min(empirical_stds),
            'empirical_perturbation_std_max': np.max(empirical_stds),
            'expected_perturbation_std': expected_std,
            'std_ratio': ratio,
            'total_rows_analyzed': rows_with_data_total,
        }
        results.append(agg_result)
    
    # Generate distribution plots
    print(f"\n{'='*60}")
    print("Generating before/after distribution plots...")
    print(f"{'='*60}\n")
    
    for sensor_key, col_name in sensor_mappings.items():
        if sensor_key not in uncertainty_params:
            continue
        
        orig_vals = np.array(sensor_original_values[sensor_key])
        pert_vals = np.array(sensor_perturbed_values[sensor_key])
        
        if len(orig_vals) < 10 or len(pert_vals) < 10:
            print(f"  X {sensor_key}: Insufficient data for plots (orig={len(orig_vals)}, pert={len(pert_vals)})")
            continue
        
        print(f"  Plotting {sensor_key}:")
        print(f"    Original: n={len(orig_vals)}, mean={np.mean(orig_vals):.6f}, std={np.std(orig_vals):.6f}, range=[{np.min(orig_vals):.6f}, {np.max(orig_vals):.6f}]")
        print(f"    Perturbed: n={len(pert_vals)}, mean={np.mean(pert_vals):.6f}, std={np.std(pert_vals):.6f}, range=[{np.min(pert_vals):.6f}, {np.max(pert_vals):.6f}]")
        print(f"    Mean shift: {np.mean(pert_vals) - np.mean(orig_vals):.6f}")
        print(f"    Std ratio: {np.std(pert_vals) / np.std(orig_vals):.3f}")
        
        # Generate before/after plots
        plot_stats = plot_before_after_distributions(
            sensor_key, 
            orig_vals, 
            pert_vals, 
            uncertainty_params[sensor_key], 
            str(output_dir)
        )
        
        # Generate diagnostic statistics
        diag_stats = plot_distribution_diagnostics(
            sensor_key,
            orig_vals,
            pert_vals,
            uncertainty_params[sensor_key],
            str(output_dir)
        )
    
    print(f"\nOK: Distribution plots saved to: {output_dir}\n")
    
    # Summary report
    print(f"{'='*60}")
    print(f"Summary: {len(results)} sensors verified")
    print(f"{'='*60}\n")
    
    if results:
        results_df = pd.DataFrame(results)
        report_path = os.path.join(str(output_dir), "perturbation_magnitude_summary.csv")
        results_df.to_csv(report_path, index=False)
        print(f"\nOK: Saved summary report: {report_path}")
        print("\nPer-Row Perturbation Magnitude Verification Results:")
        summary_cols = ['sensor', 'n_segments', 'n_calibration_points', 'components', 'empirical_perturbation_std_mean', 'expected_perturbation_std', 'std_ratio', 'total_rows_analyzed']
        available_cols = [col for col in summary_cols if col in results_df.columns]
        print(results_df[available_cols].to_string(index=False))
        
        print("\nInterpretation:")
        print(f"- Empirical std: Average of per-row perturbation stds (how much individual rows vary across {N_MC_REPLICATES} replicates)")
        print("- Expected std: Offset variance from Correction1 distribution (drawn once per replicate)")
        print("- Expected = offset_std (on normalized scale, offset-only model)")
        print(f"- std_ratio = empirical / expected (should be ~1.0 if model is correct)")
        print(f"  - OK: {RATIO_OK_MIN} to {RATIO_OK_MAX}")
        print(f"  - WARN: {RATIO_WARN_MIN} to {RATIO_WARN_MAX} (outside OK range)")
        print(f"  - FAIL: outside WARN range")
        print("- std_ratio >> 1.0 = perturbations larger than expected")
        print("- std_ratio << 1.0 = perturbations smaller than expected")
    else:
        print("WARNING: No valid sensor data found in any segments to compare.")
    
    print(f"\nOK: Verification complete. Results saved to: {output_dir}")

if __name__ == '__main__':
    main()

