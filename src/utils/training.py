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

def aggregation_slug(input_aggregation) -> str:
    """Filename-safe token for a window representation, or "" for the raw window.

    Anything cached per model has to be keyed by this as well as by the dataset.
    Hyperparameters tuned on a flattened 7381-column window are not meaningful on a
    44-column summary of the same data, and a cache that ignores the difference hands
    whichever representation ran first to all of them.
    """
    text = str(input_aggregation or "none").strip().lower()
    if text in ("", "none"):
        return ""
    return "_" + re.sub(r"[^a-z0-9]+", "-", text).strip("-")


_DEFAULT_WINDOW_STATS = ("mean", "min", "max", "std")

def _nanslope(a, axis=0):
    """Least-squares slope per column, in units per row, ignoring missing values.

    Direction of change within a block is physically meaningful for these predictors --
    falling pressure and rising temperature are not the same state as their reverses --
    and no combination of mean, min, max and std recovers it.
    """
    a = np.asarray(a, dtype=float)
    if axis != 0:
        a = np.moveaxis(a, axis, 0)
    n_rows = a.shape[0]
    if n_rows < 2:
        return np.full(a.shape[1:], np.nan)
    x = np.arange(n_rows, dtype=float)[:, None]
    valid = np.isfinite(a)
    counts = valid.sum(axis=0)
    xf = np.where(valid, x, np.nan)
    af = np.where(valid, a, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = np.nanmean(xf, axis=0)
        a_mean = np.nanmean(af, axis=0)
        dx = xf - x_mean
        da = af - a_mean
        num = np.nansum(dx * da, axis=0)
        den = np.nansum(dx * dx, axis=0)
        slope = np.where(den > 0, num / den, np.nan)
    return np.where(counts >= 2, slope, np.nan)


_WINDOW_STAT_FUNCS = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "median": np.nanmedian,
    "sum": np.nansum,
    "slope": _nanslope,
    "last": lambda a, axis: np.take(a, -1, axis=axis),
    "first": lambda a, axis: np.take(a, 0, axis=axis),
}


def parse_input_aggregation(spec):
    """Parse an ``input_aggregation`` string into ``(mode, params)``.

    Accepted forms::

        none                      the raw window, one row per timestep
        mean                      one row, the per-predictor mean
        lag:<hours>               rolling mean over consecutive <hours>-long
                                  intervals, anchored at the end of the window
        stats:<n>[:<list>]        <n> equal blocks, each summarised by the named
                                  statistics (default mean,min,max,std)
        stats:<n>h[:<list>]       blocks of <n> hours instead of a fixed count

    Prefer the ``h`` form. Window length is not the same for every target -- 671 rows
    for pH, Colour, Turbidity and intestinal enterococci, 167 for the other ten -- so a
    fixed block *count* means daily blocks on the long windows and six-hourly blocks on
    the short ones, and the same configuration name then describes two different
    representations. An interval keeps the meaning fixed and lets the block count
    follow the window.

    A 671-hour window of 11 predictors is 7381 columns once flattened, which is the
    shape XGBoost actually receives against roughly 30 distinct training samples.
    ``mean`` collapses that to 11 and discards every temporal feature of the window.
    ``lag`` and ``stats`` sit between the two: ``lag:24`` keeps 28 daily timesteps,
    ``stats:28`` keeps 28 blocks summarised four ways, so a predictor's level, spread
    and extremes within each block survive. For a sequence model the result is still a
    sequence, just a shorter one, which is what makes the 671-step positional
    embedding affordable.
    """
    text = str(spec or "none").strip().lower()
    if text in ("", "none"):
        return "none", {}
    if text == "mean":
        return "stats", {"blocks": 1, "stats": ("mean",)}

    head, _, rest = text.partition(":")
    if head == "lag":
        step = int(rest) if rest else 1
        if step < 1:
            raise ValueError(f"input_aggregation 'lag' needs an interval >= 1, got {rest!r}")
        return "lag", {"step": step}
    if head == "stats":
        parts = [x for x in rest.split(":") if x != ""]
        spec_n = parts[0] if parts else "1"
        by_interval = spec_n.endswith("h")
        n = int(spec_n[:-1]) if by_interval else int(spec_n)
        if n < 1:
            raise ValueError(f"input_aggregation 'stats' needs a positive size, got {parts!r}")
        names = tuple(parts[1].split(",")) if len(parts) > 1 else _DEFAULT_WINDOW_STATS
        unknown = [n for n in names if n not in _WINDOW_STAT_FUNCS]
        if unknown:
            raise ValueError(
                f"Unknown window statistic(s) {unknown}; "
                f"available: {sorted(_WINDOW_STAT_FUNCS)}"
            )
        key = "block_hours" if by_interval else "blocks"
        return "stats", {key: n, "stats": names}

    raise ValueError(
        f"Unrecognised input_aggregation {spec!r}. Use none, mean, lag:<step> "
        "or stats:<blocks>[:<stats>]."
    )


def _stats_block_count(n_rows: int, params: dict) -> int:
    """How many blocks a window of *n_rows* is divided into under *params*."""
    n_rows = max(1, int(n_rows))
    if "block_hours" in params:
        width = max(1, int(params["block_hours"]))
        return max(1, int(np.ceil(n_rows / width)))
    return max(1, min(int(params.get("blocks", 1)), n_rows))


def _lag_indices(n_rows: int, step: int) -> np.ndarray:
    """Row indices for lag subsampling, anchored at the end of the window.

    Sampling forward from the first row leaves a remainder at the end: over 671 rows at
    a 24-hour step the last row taken is 648, so the most recent 22 hours -- the data
    closest to the forecast origin, and the most informative for it -- are discarded.
    A sequence model makes that worse, reading the final timestep as its summary of the
    window. Anchoring at the end instead puts the remainder at the oldest edge, where
    dropping it costs least, and guarantees the final row is always included.
    """
    n_rows = int(n_rows)
    step = max(1, int(step))
    return np.arange(n_rows - 1, -1, -step, dtype=int)[::-1]


def _rolling_interval_mean(input_seq: np.ndarray, step: int) -> np.ndarray:
    """Mean of each consecutive *step*-row interval, anchored at the window end."""
    n_rows = input_seq.shape[0]
    edges = _lag_indices(n_rows, step)
    rows = []
    with np.errstate(invalid="ignore"):
        for end in edges:
            start = max(0, int(end) - int(step) + 1)
            rows.append(np.nanmean(input_seq[start:int(end) + 1, :], axis=0))
    return np.asarray(rows, dtype=float)


def reduced_window_shape(n_rows, n_features, spec):
    """The ``(timesteps, features)`` a window becomes under *spec*.

    The transformer reads its ``seq_len`` and ``input_dim`` from configuration rather
    than from the data, so anything that reshapes the window has to be reflected here
    or the model is built for the wrong input.
    """
    mode, params = parse_input_aggregation(spec)
    n_rows = int(n_rows)
    n_features = int(n_features)
    if mode == "none":
        return n_rows, n_features
    if mode == "lag":
        return len(_lag_indices(n_rows, params["step"])), n_features
    return _stats_block_count(n_rows, params), n_features * len(params["stats"])


def reduce_input_window(input_seq, spec):
    """Apply *spec* to one ``(timesteps, features)`` window."""
    mode, params = parse_input_aggregation(spec)
    if mode == "none":
        return input_seq
    if mode == "lag":
        # The mean over each interval, not the single value at its end. Several
        # predictors vary strongly through the day, so one reading per day samples a
        # fixed hour and aliases that variation into whatever phase the window happens
        # to start on; the interval mean does not.
        return _rolling_interval_mean(input_seq, params["step"])

    n_rows, n_features = input_seq.shape
    blocks = _stats_block_count(n_rows, params)
    names = params["stats"]
    # Contiguous, near-equal blocks in time order; np.array_split handles a window
    # length that does not divide evenly without dropping the remainder.
    chunks = np.array_split(input_seq, blocks, axis=0)
    out = np.empty((blocks, n_features * len(names)), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for bi, chunk in enumerate(chunks):
            for si, name in enumerate(names):
                values = _WINDOW_STAT_FUNCS[name](chunk, axis=0)
                # A predictor's statistics stay adjacent, so feature importances read
                # as "this predictor, this statistic" rather than interleaved.
                out[bi, si::len(names)] = values
    return out


def load_samples(directory, input_columns, output_columns, input_rows, output_rows, file_list=None,
                 fault_tolerant=False, source=None, input_aggregation='none', drop_report=None):
    """Load windowed samples, recording why each rejected sample was rejected.

    Every ``continue`` below discards a sample, and until these were counted the
    loss was invisible: a subset containing one partial-coverage predictor could
    cost a model 17 of 22 evaluation samples with nothing written down, so the
    effect could only be inferred afterwards by comparing subsets. Pass a dict as
    *drop_report* to receive the tally, including which predictor columns made
    samples unusable. A non-empty tally is also summarized on stdout, so runs
    that do not opt in still leave a record.

    Behaviour is unchanged: the same samples are loaded and rejected as before.
    """
    samples = []
    report = {
        "n_considered": 0,
        "n_loaded": 0,
        "dropped_missing_columns": 0,
        "dropped_too_few_rows": 0,
        "dropped_all_nan_predictor": 0,
        "dropped_nan_output": 0,
        "dropped_nan_input": 0,
        "nan_input_columns": {},
        "dropped_files": [],
    }

    def _tally_columns(mask, reason, filename):
        """Attribute a drop to the predictor columns responsible for it."""
        for col in (c for c, bad in zip(input_columns, mask) if bad):
            report["nan_input_columns"][col] = report["nan_input_columns"].get(col, 0) + 1
        report["dropped_files"].append((filename, reason))

    if source is not None:
        with open(source) as f:
            file_list = [line.strip() for line in f]
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv"):
            continue
        if file_list is not None and filename not in file_list:
            continue  # Skip files not in the provided list
        report["n_considered"] += 1
        df = pd.read_csv(os.path.join(directory, filename))
        if not set(input_columns + output_columns).issubset(df.columns):
            report["dropped_missing_columns"] += 1
            missing = [c for c in input_columns if c not in df.columns]
            _tally_columns([c in missing for c in input_columns], "missing_columns", filename)
            continue  # skip files with missing columns
        if len(df) < input_rows.stop:
            report["dropped_too_few_rows"] += 1
            report["dropped_files"].append((filename, "too_few_rows"))
            continue  # skip files without enough rows
        input_seq = df.iloc[input_rows, :][input_columns].values
        _agg_mode, _agg_params = parse_input_aggregation(input_aggregation)
        if _agg_mode == "stats":
            # A statistic over an all-NaN predictor is NaN however it is computed, so
            # the sample is unusable; say so here rather than letting it surface later
            # as an unexplained NaN input.
            predictor_all_nan = np.all(np.isnan(input_seq), axis=0)
            if np.any(predictor_all_nan):
                report["dropped_all_nan_predictor"] += 1
                _tally_columns(predictor_all_nan, "all_nan_predictor", filename)
                continue
        if _agg_mode != "none":
            input_seq = reduce_input_window(input_seq, input_aggregation)
        # Handle output_rows as either a list of indices or a starting index for slicing
        if isinstance(output_rows, list):
            output_seq = df.iloc[output_rows, :][output_columns].values
        else:
            output_seq = df.iloc[output_rows:, :][output_columns].values
        # Always skip samples with NaN in outputs/labels (no model can train with these)
        if np.isnan(output_seq).any():
            report["dropped_nan_output"] += 1
            report["dropped_files"].append((filename, "nan_output"))
            continue
        # Only skip samples with NaN in inputs when fault_tolerant=False
        if not fault_tolerant and np.isnan(input_seq).any():
            report["dropped_nan_input"] += 1
            _tally_columns(np.any(np.isnan(input_seq), axis=0), "nan_input", filename)
            continue
        samples.append((input_seq, output_seq, filename))

    report["n_loaded"] = len(samples)
    n_dropped = report["n_considered"] - report["n_loaded"]
    if n_dropped > 0:
        reasons = ", ".join(
            f"{k.replace('dropped_', '')}={report[k]}"
            for k in ("dropped_missing_columns", "dropped_too_few_rows",
                      "dropped_all_nan_predictor", "dropped_nan_output", "dropped_nan_input")
            if report[k]
        )
        culprits = sorted(report["nan_input_columns"].items(), key=lambda kv: -kv[1])[:4]
        culprit_txt = ("; worst predictors: "
                       + ", ".join(f"{c} ({n})" for c, n in culprits)) if culprits else ""
        print(f"[INFO] Samples loaded: {report['n_loaded']} of {report['n_considered']}; "
              f"dropped {n_dropped} ({reasons}){culprit_txt}")
    else:
        print(f"[INFO] Samples loaded: {report['n_loaded']} of {report['n_considered']}; no drops")

    if isinstance(drop_report, dict):
        drop_report.update(report)
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
             min_test_independent=None, allow_rebalance=True):
    ## If specified, reuse a train/test split previously written to file.
    train_samples = []
    test_samples = []
    sample_dir = Path(data_dir, sample_subdir)

    if reuse_split:
        explicit_source = split_source is not None
        try:
            if split_source is None:
                split_source = Path(data_dir, "forecasts", forecast_name)
            split_source = Path(split_source)
            own_dir = Path(data_dir, "forecasts", forecast_name)

            # A reused split may come from a different run's directory. Copy it into
            # this run's own directory and work from the copy, for two reasons: the
            # run has to carry its own split files because that is where evaluation
            # looks for them, and a rebalance below rewrites these files -- which
            # must never reach back into the run the split was borrowed from.
            own_dir.mkdir(parents=True, exist_ok=True)
            train_file = own_dir / "train_files.txt"
            test_file = own_dir / "test_files.txt"
            if split_source.resolve() != own_dir.resolve():
                train_file.write_text(
                    (split_source / "train_files.txt").read_text(encoding="utf-8"), encoding="utf-8")
                test_file.write_text(
                    (split_source / "test_files.txt").read_text(encoding="utf-8"), encoding="utf-8")

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
                if status == "rebalanced" and not allow_rebalance:
                    # Rebalancing moves segments from train into test. When the split
                    # is pinned so that every model is scored on the same segments,
                    # that would both break the shared test set and put segments the
                    # model trained on into its own evaluation. Reject the variant
                    # instead: not meeting the minimum is a property of the feature
                    # subset, and it must be visible rather than repaired in place.
                    raise SampleComplianceError(
                        reason="rebalance_forbidden_on_pinned_split",
                        message=(
                            "Reused split is pinned but does not meet "
                            f"min_test_independent={int(min_test_independent)} "
                            f"(test_independent={before_indep}); rebalancing would move "
                            "training segments into the shared test set."
                        ),
                        context={
                            "split_mode": "reuse_pinned",
                            "split_source": str(split_source),
                            "test_independent": int(before_indep),
                            "target_min_independent": int(min_test_independent),
                        },
                    )
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
            # A caller that named a split_source asked for that exact split; carrying
            # on with no samples would report a different evaluation set as though it
            # were the requested one.
            if explicit_source:
                raise FileNotFoundError(
                    f"Could not reuse the split at {split_source}: {e}"
                ) from e
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
            print("\n[WARN] Monte Carlo replicates detected!")
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