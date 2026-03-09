import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SampleComplianceError(RuntimeError):
    """Raised when sweep split/sample-count compliance requirements are not met."""

    def __init__(self, reason: str, message: str, context: dict | None = None):
        super().__init__(message)
        self.reason = str(reason)
        self.context = dict(context or {})

def load_samples(directory, input_columns, output_columns, input_rows, output_rows, file_list=None,
                 fault_tolerant=False, source=None, input_aggregation='none'):
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
            continue  # skip files without enough rows
        input_seq = df.iloc[input_rows, :][input_columns].values
        if str(input_aggregation).lower() == 'mean':
            # Require at least one finite value per predictor; otherwise sample is unusable.
            predictor_all_nan = np.all(np.isnan(input_seq), axis=0)
            if np.any(predictor_all_nan):
                continue
            with np.errstate(invalid='ignore'):
                input_seq = np.nanmean(input_seq, axis=0, keepdims=True)
        # Handle output_rows as either a list of indices or a starting index for slicing
        if isinstance(output_rows, list):
            output_seq = df.iloc[output_rows, :][output_columns].values
        else:
            output_seq = df.iloc[output_rows:, :][output_columns].values
        # Always skip samples with NaN in outputs/labels (no model can train with these)
        if np.isnan(output_seq).any():
            continue
        # Only skip samples with NaN in inputs when fault_tolerant=False
        if not fault_tolerant and np.isnan(input_seq).any():
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


def _sample_valid_count(sample):
    """Count non-NaN values across both inputs and targets for one sample."""
    input_seq, output_seq = sample[0], sample[1]
    input_valid = int(np.count_nonzero(~np.isnan(input_seq)))
    output_valid = int(np.count_nonzero(~np.isnan(output_seq)))
    return input_valid + output_valid


def _input_nan_fraction(sample):
    """Compute NaN fraction for predictor values of one sample."""
    input_seq = np.asarray(sample[0], dtype=float)
    total_values = int(input_seq.size)
    if total_values == 0:
        return 1.0
    finite_values = int(np.count_nonzero(np.isfinite(input_seq)))
    return 1.0 - (finite_values / total_values)


def _filter_samples_by_nan_tolerance(samples, nan_tolerance):
    """Keep only samples whose predictor NaN fraction is <= nan_tolerance."""
    filtered = []
    dropped = 0

    for sample in samples:
        if _input_nan_fraction(sample) <= nan_tolerance:
            filtered.append(sample)
        else:
            dropped += 1

    print(
        f"NaN pre-filter (<= {nan_tolerance:.3f} NaN fraction): "
        f"kept {len(filtered)}/{len(samples)} samples, dropped {dropped}"
    )
    return filtered


def _split_index_from_cumulative_valid_counts(items, count_fn, train_fraction):
    """
    Compute temporal split index by cutting cumulative valid_count at
    train_fraction * total_valid_count.

    Returns (split_idx, total_valid, train_valid).
    """
    if len(items) == 0:
        return 0, 0, 0

    valid_counts = [max(0, int(count_fn(item))) for item in items]
    total_valid = int(np.sum(valid_counts))

    if total_valid > 0:
        cutoff = float(train_fraction) * total_valid
        cumulative = np.cumsum(valid_counts)
        split_idx = int(np.searchsorted(cumulative, cutoff, side='left') + 1)
    else:
        split_idx = int(len(items) * float(train_fraction))

    # Keep both sets non-empty whenever possible
    if len(items) > 1:
        split_idx = max(1, min(len(items) - 1, split_idx))
    else:
        split_idx = len(items)

    train_valid = int(np.sum(valid_counts[:split_idx]))
    return split_idx, total_valid, train_valid


def _base_sample_id(name: str) -> str:
    """Collapse MC replicate suffixes so counts reflect independent raw samples."""
    filename = Path(str(name)).name
    return re.sub(r"_mc_\d+(?=\.csv$)", "", filename)


def _independent_count(names: list[str]) -> int:
    return len(dict.fromkeys(_base_sample_id(name) for name in names if str(name).strip()))


def _rebalance_to_min_test_independent(
    train_names: list[str],
    test_names: list[str],
    min_test_independent: int,
) -> tuple[bool, str, list[str], list[str]]:
    """Move latest independent train groups to test until minimum independent test count is met."""
    min_req = int(max(0, min_test_independent))
    if min_req <= 0:
        return False, "disabled", list(train_names), list(test_names)

    train_names = [str(n) for n in list(train_names) if str(n).strip()]
    test_names = [str(n) for n in list(test_names) if str(n).strip()]

    test_independent = _independent_count(test_names)
    if test_independent >= min_req:
        return False, "already_sufficient", train_names, test_names

    total_independent = _independent_count(train_names + test_names)
    if total_independent < min_req:
        return True, "insufficient_total", train_names, test_names

    test_ids = set(_base_sample_id(n) for n in test_names)
    train_ids_order = []
    seen_train_ids = set()
    for name in train_names:
        gid = _base_sample_id(name)
        if gid not in seen_train_ids:
            seen_train_ids.add(gid)
            train_ids_order.append(gid)

    moved_ids = []
    for gid in reversed(train_ids_order):
        if gid in test_ids:
            continue
        moved_ids.append(gid)
        test_ids.add(gid)
        if len(test_ids) >= min_req:
            break

    if len(test_ids) < min_req:
        return True, "insufficient_total", train_names, test_names

    moved_set = set(moved_ids)
    moved_names = [name for name in train_names if _base_sample_id(name) in moved_set]
    new_train = [name for name in train_names if _base_sample_id(name) not in moved_set]
    # Keep temporal ordering by prepending moved train tail ahead of existing test rows.
    new_test = moved_names + test_names
    return False, "rebalanced", new_train, new_test

def write_config(config, data_dir, forecast_name, model_name, config_name='model_config.json'):
    ## Write model configuration dictionary to file so it can be re-run and re-used for other model types
    filepath = Path(data_dir, 'forecasts', forecast_name, model_name, config_name)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f)

def splitter(data_dir, forecast_name, input_columns, input_rows, output_columns, output_rows, fault_tolerant=True,
             reuse_split=True, split_source=None, split_type='random', test_size=0.3, random_state=10,
             sample_subdir='samples', nan_tolerance=None, input_aggregation='none',
             min_test_independent=None):
    ## If specified, reuse a train/test split previously written to file.
    train_samples = []
    test_samples = []
    sample_dir = Path(data_dir, sample_subdir)

    if reuse_split:
        try:
            if split_source is None:
                split_source = Path(data_dir, "forecasts", forecast_name)
            split_source = Path(split_source)
            train_file = split_source / "train_files.txt"
            test_file = split_source / "test_files.txt"
            train_file_names = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            test_file_names = [line.strip() for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            train_samples = load_samples(sample_dir, input_columns=input_columns,
                                         output_columns=output_columns,
                                         input_rows=input_rows, output_rows=output_rows, file_list=None,
                                         fault_tolerant=fault_tolerant, source=train_file,
                                         input_aggregation=input_aggregation)
            test_samples = load_samples(sample_dir, input_columns=input_columns,
                                        output_columns=output_columns,
                                        input_rows=input_rows, output_rows=output_rows, file_list=None,
                                        fault_tolerant=fault_tolerant, source=test_file,
                                        input_aggregation=input_aggregation)

            if min_test_independent is not None:
                train_valid_names = {str(sample[2]) for sample in train_samples}
                test_valid_names = {str(sample[2]) for sample in test_samples}
                train_ordered_valid = [n for n in train_file_names if n in train_valid_names]
                test_ordered_valid = [n for n in test_file_names if n in test_valid_names]
                skip_eval, status, new_train_names, new_test_names = _rebalance_to_min_test_independent(
                    train_ordered_valid,
                    test_ordered_valid,
                    int(min_test_independent),
                )
                before_indep = _independent_count(test_ordered_valid)
                after_indep = _independent_count(new_test_names)
                if status == "rebalanced":
                    print(
                        f"Rebalanced reused split for min independent test samples: "
                        f"test_independent {before_indep} -> {after_indep} "
                        f"(target={int(min_test_independent)})."
                    )
                    train_file.write_text("\n".join(new_train_names) + ("\n" if new_train_names else ""), encoding="utf-8")
                    test_file.write_text("\n".join(new_test_names) + ("\n" if new_test_names else ""), encoding="utf-8")
                    train_samples = load_samples(sample_dir, input_columns=input_columns,
                                                 output_columns=output_columns,
                                                 input_rows=input_rows, output_rows=output_rows, file_list=None,
                                                 fault_tolerant=fault_tolerant, source=train_file,
                                                 input_aggregation=input_aggregation)
                    test_samples = load_samples(sample_dir, input_columns=input_columns,
                                                output_columns=output_columns,
                                                input_rows=input_rows, output_rows=output_rows, file_list=None,
                                                fault_tolerant=fault_tolerant, source=test_file,
                                                input_aggregation=input_aggregation)
                elif skip_eval:
                    raise SampleComplianceError(
                        reason="insufficient_total_independent",
                        message=(
                            "Reused split cannot satisfy min independent test samples "
                            f"(test_independent={before_indep}, target={int(min_test_independent)})."
                        ),
                        context={
                            "split_mode": "reuse",
                            "split_source": str(split_source),
                            "test_independent": int(before_indep),
                            "target_min_independent": int(min_test_independent),
                        },
                    )

            print(f'Reused split in {split_source}. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')
        except SampleComplianceError:
            raise
        except Exception as e:
            print(f"No previous split available for reuse: {e}")
    else:
        ## Generate a new split.
        samples = load_samples(sample_dir, input_columns=input_columns,
                               output_columns=output_columns, input_rows=input_rows, output_rows=output_rows,
                               fault_tolerant=fault_tolerant, input_aggregation=input_aggregation)

        if fault_tolerant and nan_tolerance is not None:
            samples = _filter_samples_by_nan_tolerance(samples, nan_tolerance)
            if len(samples) < 2:
                raise ValueError(
                    "Not enough samples remain after NaN pre-filtering to create train/test split. "
                    "Increase nan_tolerance, generate more samples, or reduce test_size."
                )
        # print('samples', samples)
        
        # Detect Monte Carlo replicates and adjust split strategy if needed
        is_mc_dataset, segment_groups = detect_mc_replicates(samples)
        if is_mc_dataset:
            print("\n⚠️  Monte Carlo replicates detected!")
            print("   Enforcing temporal split to prevent data leakage.")
            print("   All replicates of each segment will stay together in train/test.\n")
            split_type = 'temporal'  # Force temporal split for MC datasets
        
        if split_type == 'temporal':
            ## Time-based split with MC-aware grouping if needed
            train_fraction = 1 - test_size
            if is_mc_dataset:
                # Group samples by segment number
                segment_groups_list = group_samples_by_segment(samples)
                split_idx, total_valid, train_valid = _split_index_from_cumulative_valid_counts(
                    segment_groups_list,
                    lambda group_item: sum(_sample_valid_count(s) for s in group_item[1]),
                    train_fraction,
                )
                
                # Flatten the groups back to samples
                train_samples = []
                test_samples = []
                for i, (seg_num, seg_samples) in enumerate(segment_groups_list):
                    if i < split_idx:
                        train_samples.extend(seg_samples)
                    else:
                        test_samples.extend(seg_samples)

                test_valid = total_valid - train_valid
                achieved_train_frac = (train_valid / total_valid) if total_valid > 0 else 0.0
                
                print(f'Temporal split (MC-aware). Training set: {len(train_samples)} samples. '
                      f'Test set: {len(test_samples)} samples')
                print(f'  Valid data coverage (input+target non-NaN): '
                      f'train={train_valid}, test={test_valid}, total={total_valid}, '
                      f'train_fraction={achieved_train_frac:.3f} (target={train_fraction:.3f})')
            else:
                # Standard temporal split without MC grouping
                samples_sorted = sorted(samples, key=extract_index)
                split_idx, total_valid, train_valid = _split_index_from_cumulative_valid_counts(
                    samples_sorted,
                    _sample_valid_count,
                    train_fraction,
                )
                train_samples = samples_sorted[:split_idx]
                test_samples = samples_sorted[split_idx:]

                test_valid = total_valid - train_valid
                achieved_train_frac = (train_valid / total_valid) if total_valid > 0 else 0.0
                print(f'Time-based split. Training set: {len(train_samples)} samples. '
                      f'Test set: {len(test_samples)} samples')
                print(f'  Valid data coverage (input+target non-NaN): '
                      f'train={train_valid}, test={test_valid}, total={total_valid}, '
                      f'train_fraction={achieved_train_frac:.3f} (target={train_fraction:.3f})')
        else:
            ## Random shuffle
            train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
            print(f'Randomized split. Training set: {len(train_samples)} samples. Test set: {len(test_samples)} samples')

        ## Write new split to file, to enable error checking and reuse
        file1 = Path(data_dir, "forecasts", forecast_name, "train_files.txt")
        file1.parent.mkdir(parents=True, exist_ok=True)
        file2 = Path(data_dir, "forecasts", forecast_name, "test_files.txt")

        train_names = [str(s[2]) for s in train_samples]
        test_names = [str(s[2]) for s in test_samples]
        if min_test_independent is not None:
            skip_eval, status, train_names, test_names = _rebalance_to_min_test_independent(
                train_names,
                test_names,
                int(min_test_independent),
            )
            before_indep = _independent_count([str(s[2]) for s in test_samples])
            after_indep = _independent_count(test_names)
            if status == "rebalanced":
                print(
                    f"Rebalanced new split for min independent test samples: "
                    f"test_independent {before_indep} -> {after_indep} "
                    f"(target={int(min_test_independent)})."
                )
            elif skip_eval:
                raise SampleComplianceError(
                    reason="insufficient_total_independent",
                    message=(
                        "New split cannot satisfy min independent test samples "
                        f"(test_independent={before_indep}, target={int(min_test_independent)})."
                    ),
                    context={
                        "split_mode": "new",
                        "forecast_name": str(forecast_name),
                        "data_dir": str(data_dir),
                        "test_independent": int(before_indep),
                        "target_min_independent": int(min_test_independent),
                    },
                )

        with open(file1, "w") as f:
            f.writelines(f"{name}\n" for name in train_names)
        with open(file2, "w") as f:
            f.writelines(f"{name}\n" for name in test_names)

        train_samples = load_samples(sample_dir, input_columns=input_columns,
                                     output_columns=output_columns,
                                     input_rows=input_rows, output_rows=output_rows, file_list=None,
                                     fault_tolerant=fault_tolerant, source=file1,
                                     input_aggregation=input_aggregation)
        test_samples = load_samples(sample_dir, input_columns=input_columns,
                                    output_columns=output_columns,
                                    input_rows=input_rows, output_rows=output_rows, file_list=None,
                                    fault_tolerant=fault_tolerant, source=file2,
                                    input_aggregation=input_aggregation)
    return train_samples, test_samples