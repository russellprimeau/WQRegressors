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
            continue  # Skip files not in the provided list
        df = pd.read_csv(os.path.join(directory, filename))
        if not set(input_columns + output_columns).issubset(df.columns):
            continue  # skip files with missing columns
        if len(df) < input_rows.stop:
            print(f"Sample {filename} skipped — not enough rows ({len(df)} < {input_rows.stop})")
            continue  # skip files without enough rows

        input_seq = df.iloc[input_rows, :][input_columns].values
        output_seq = df.iloc[output_rows:, :][output_columns].values
        if not fault_tolerant:
            if np.isnan(input_seq).any() or np.isnan(output_seq).any():
                print(f"Sample {filename} skipped - contains NaN values")
                continue  # skip invalid samples
        samples.append((input_seq, output_seq, filename))
    print("Samples loaded")
    return samples

def extract_index(sample):
    # Extract index value from sample name for ordering samples
    filename = os.path.basename(sample[-1])
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def write_config(config, data_dir, forecast_name, model_name, config_name='model_config.json'):
    ## Write model configuration dictionary to file so it can be re-run and re-used for other model types
    filepath = Path(data_dir, 'forecasts', forecast_name, model_name, config_name)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f)

def splitter(data_dir, forecast_name, input_columns, input_rows, output_columns, output_rows, fault_tolerant=True,
             reuse_split=True, split_source=None, split_type='random', test_size=0.2, random_state=10):
    ## If specified, reuse a train/test split previously written to file.
    train_samples = []
    test_samples = []
    if reuse_split:
        try:
            if split_source is None:
                split_source = Path(data_dir, "forecasts", forecast_name)
            train_samples = load_samples(Path(data_dir, "samples"), input_columns=input_columns,
                                         output_columns=output_columns,
                                         input_rows=input_rows, output_rows=output_rows, file_list=None,
                                         fault_tolerant=fault_tolerant, source=Path(split_source, "train_files.txt"))
            test_samples = load_samples(Path(data_dir, "samples"), input_columns=input_columns,
                                        output_columns=output_columns,
                                        input_rows=input_rows, output_rows=output_rows, file_list=None,
                                        fault_tolerant=fault_tolerant, source=Path(split_source, "test_files.txt"))
            print(f'Reused split in {split_source}. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')
        except Exception as e:
            print(f"No previous split available for reuse: {e}")
    else:
        ## Generate a new split.
        samples = load_samples(os.path.join(data_dir, 'samples'), input_columns=input_columns,
                               output_columns=output_columns, input_rows=input_rows, output_rows=output_rows,
                               fault_tolerant=fault_tolerant)
        if split_type == 'temporal':
            ## Time-based split
            samples_sorted = sorted(samples, key=extract_index)
            split_idx = int(len(samples_sorted) * (1 - test_size))  # Compute split point
            train_samples = samples_sorted[:split_idx]
            test_samples = samples_sorted[split_idx:]
            print(
                f'Time-based split. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')
        else:
            ## Random shuffle
            train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
            print(f'Randomized split. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')

        ## Write new split to file, to enable error checking and reuse
        file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
        with open(file1, "w") as f:
            f.writelines(f"{s[2]}\n" for s in train_samples)
        file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")
        with open(file2, "w") as f:
            f.writelines(f"{s[2]}\n" for s in test_samples)
    return train_samples, test_samples