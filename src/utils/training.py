import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_samples(directory, input_columns, output_columns, input_rows, output_rows, file_list=None,
                 fault_tolerant=False, source=None):
    samples = []

    if source is not None:
        with open(source) as f:
            file_list = [line.strip() for line in f]
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv"):
            continue
        if file_list is not None and filename not in file_list:
            print(f"Sample {filename} skipped - not in list")
            continue  # Skip files not in the provided list
        df = pd.read_csv(os.path.join(directory, filename))
        if not set(input_columns + output_columns).issubset(df.columns):
            print(f"Sample {filename} skipped - contains missing columns")
            print('Contains only:', df.columns)
            continue  # skip files with missing columns
        if len(df) < input_rows.stop:
            print(f"Sample {filename} skipped — not enough rows ({len(df)} < {input_rows.stop})")
            continue  # skip files without enough rows
        input_seq = df.iloc[input_rows, :][input_columns].values
        # Handle output_rows as either a list of indices or a starting index for slicing
        if isinstance(output_rows, list):
            output_seq = df.iloc[output_rows, :][output_columns].values
        else:
            output_seq = df.iloc[output_rows:, :][output_columns].values
        # Always skip samples with NaN in outputs/labels (no model can train with these)
        if np.isnan(output_seq).any():
            print(f"Sample {filename} skipped - contains NaN in output/labels")
            continue
        # Only skip samples with NaN in inputs when fault_tolerant=False
        if not fault_tolerant and np.isnan(input_seq).any():
            print(f"Sample {filename} skipped - contains NaN in input features")
            continue
        samples.append((input_seq, output_seq, filename))
    print("Samples loaded")
    return samples

def extract_index(sample):
    # Extract index value from sample name for ordering samples
    filename = os.path.basename(sample[-1])
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def detect_mc_replicates(samples):
    """
    Detect if samples contain Monte Carlo replicates (files with _mc_ in name).
    Returns (is_mc_dataset, segment_groups) where:
    - is_mc_dataset: bool indicating presence of MC replicates
    - segment_groups: dict mapping segment_number -> list of samples for that segment
    """
    segment_groups = {}
    has_mc = False
    
    for sample in samples:
        filename = os.path.basename(sample[-1])
        
        # Check if this is an MC replicate file
        if '_mc_' in filename:
            has_mc = True
        
        # Extract segment number (e.g., "segment_0001_mc_005.csv" -> 1)
        match = re.search(r'segment_(\d+)', filename)
        if match:
            segment_num = int(match.group(1))
            if segment_num not in segment_groups:
                segment_groups[segment_num] = []
            segment_groups[segment_num].append(sample)
    
    return has_mc, segment_groups

def group_samples_by_segment(samples):
    """
    Group samples by segment number to keep MC replicates together.
    Returns list of (segment_number, [samples]) tuples, sorted by segment number.
    """
    segment_groups = {}
    
    for sample in samples:
        filename = os.path.basename(sample[-1])
        # Extract segment number (e.g., "segment_0001_mc_005.csv" -> 1)
        match = re.search(r'segment_(\d+)', filename)
        if match:
            segment_num = int(match.group(1))
            if segment_num not in segment_groups:
                segment_groups[segment_num] = []
            segment_groups[segment_num].append(sample)
    
    # Return sorted by segment number (temporal order)
    return sorted(segment_groups.items(), key=lambda x: x[0])

def write_config(config, data_dir, forecast_name, model_name, config_name='model_config.json'):
    ## Write model configuration dictionary to file so it can be re-run and re-used for other model types
    filepath = Path(data_dir, 'forecasts', forecast_name, model_name, config_name)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f)

def splitter(data_dir, forecast_name, input_columns, input_rows, output_columns, output_rows, fault_tolerant=True,
             reuse_split=True, split_source=None, split_type='random', test_size=0.2, random_state=10,
             sample_subdir='samples'):
    ## If specified, reuse a train/test split previously written to file.
    train_samples = []
    test_samples = []
    sample_dir = Path(data_dir, sample_subdir)

    if reuse_split:
        try:
            if split_source is None:
                split_source = Path(data_dir, "forecasts", forecast_name)
            train_samples = load_samples(sample_dir, input_columns=input_columns,
                                         output_columns=output_columns,
                                         input_rows=input_rows, output_rows=output_rows, file_list=None,
                                         fault_tolerant=fault_tolerant, source=Path(split_source, "train_files.txt"))
            test_samples = load_samples(sample_dir, input_columns=input_columns,
                                        output_columns=output_columns,
                                        input_rows=input_rows, output_rows=output_rows, file_list=None,
                                        fault_tolerant=fault_tolerant, source=Path(split_source, "test_files.txt"))
            print(f'Reused split in {split_source}. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')
        except Exception as e:
            print(f"No previous split available for reuse: {e}")
    else:
        ## Generate a new split.
        samples = load_samples(sample_dir, input_columns=input_columns,
                               output_columns=output_columns, input_rows=input_rows, output_rows=output_rows,
                               fault_tolerant=fault_tolerant)
        print('samples', samples)
        
        # Detect Monte Carlo replicates and adjust split strategy if needed
        is_mc_dataset, segment_groups = detect_mc_replicates(samples)
        if is_mc_dataset:
            print("\n⚠️  Monte Carlo replicates detected!")
            print("   Enforcing temporal split to prevent data leakage.")
            print("   All replicates of each segment will stay together in train/test.\n")
            split_type = 'temporal'  # Force temporal split for MC datasets
        
        if split_type == 'temporal':
            ## Time-based split with MC-aware grouping if needed
            if is_mc_dataset:
                # Group samples by segment number
                segment_groups_list = group_samples_by_segment(samples)
                split_idx = int(len(segment_groups_list) * (1 - test_size))
                
                # Flatten the groups back to samples
                train_samples = []
                test_samples = []
                for i, (seg_num, seg_samples) in enumerate(segment_groups_list):
                    if i < split_idx:
                        train_samples.extend(seg_samples)
                    else:
                        test_samples.extend(seg_samples)
                
                print(f'Temporal split (MC-aware). Training set: {len(train_samples)} samples. '
                      f'Test set: {len(test_samples)} samples')
            else:
                # Standard temporal split without MC grouping
                samples_sorted = sorted(samples, key=extract_index)
                split_idx = int(len(samples_sorted) * (1 - test_size))
                train_samples = samples_sorted[:split_idx]
                test_samples = samples_sorted[split_idx:]
                print(f'Time-based split. Training set: {len(train_samples)} samples. '
                      f'Test set: {len(test_samples)} samples')
        else:
            ## Random shuffle
            train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
            print(f'Randomized split. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')

        ## Write new split to file, to enable error checking and reuse
        file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
        file1.parent.mkdir(parents=True, exist_ok=True)
        with open(file1, "w") as f:
            f.writelines(f"{s[2]}\n" for s in train_samples)
        file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
        with open(file2, "w") as f:
            f.writelines(f"{s[2]}\n" for s in test_samples)
    return train_samples, test_samples