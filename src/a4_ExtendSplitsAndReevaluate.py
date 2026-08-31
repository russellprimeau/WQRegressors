"""Re-evaluate the runs affected by the predictor gap fill, without retraining.

Filling short gaps in the predictor series (see ``interpolate_short_gaps`` in
``utils/preprocessing.py``) changes two things about an already-trained run:

  1. Segments that were discarded for containing a missing predictor value may now
     be scorable. They were never in the run's split at all, because invalid samples
     are dropped before the split is written, so recovering them means extending the
     test list -- re-running evaluation alone would not find them.
  2. Segments that *were* scored may now have different inputs, for the families that
     tolerate missing values and scored them with the gaps in place.

Only runs in one of those two situations are touched. Everything else keeps the
predictions it has.

**A recovered segment is admitted to the test set only if it falls after the run's
existing train/test boundary.** Such a segment was in neither split, so scoring it
cannot leak -- the model never saw it -- but one from the training period would stop
the split being a temporal holdout, and the comparison in the results table depends
on that. Recovered segments before the boundary are left out and counted.

Retraining is out of scope: these models were selected and fitted on the smaller
sample sets, and that is a property of the study to be disclosed, not repaired here.

Usage:
    python src/a4_ExtendSplitsAndReevaluate.py --root data/output/CV20_profilerless
    python src/a4_ExtendSplitsAndReevaluate.py --root ... --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

# Retained-run membership comes from z8, so this re-evaluates exactly the runs the
# results table can draw on and nothing else. A sweep leaves a directory behind for
# every candidate it tried during the search -- for one target that is 277 XGBoost
# directories against 12 retained configurations -- and those trials are never scored.
import z8_CommonSetMetrics as z8

REPO_ROOT = Path(__file__).resolve().parent.parent

_MC_SUFFIX = re.compile(r'_mc_\d+(?=\.csv$)')


def base_name(name: str) -> str:
    """Segment id with any Monte Carlo replicate suffix removed."""
    return _MC_SUFFIX.sub('', Path(name).name)


def seg_index(name: str) -> int:
    m = re.search(r'(\d+)', base_name(name))
    return int(m.group(1)) if m else -1


def changed_cells(old_csv: Path, new_csv: Path) -> dict:
    """{predictor: set of timestamps whose value changed or appeared}."""
    old = pd.read_csv(old_csv, parse_dates=['TIMESTAMP'],
                      encoding='utf-8', encoding_errors='replace').set_index('TIMESTAMP')
    new = pd.read_csv(new_csv, parse_dates=['TIMESTAMP'],
                      encoding='utf-8', encoding_errors='replace').set_index('TIMESTAMP')
    idx = old.index.intersection(new.index)
    old, new = old.loc[idx], new.loc[idx]
    out = {}
    for c in old.columns:
        if c not in new.columns:
            continue
        a = pd.to_numeric(old[c], errors='coerce')
        b = pd.to_numeric(new[c], errors='coerce')
        diff = (a.isna() & b.notna()) | (a.notna() & b.isna()) | \
               (a.notna() & b.notna() & (a != b))
        if diff.any():
            out[c] = set(idx[diff])
    return out


def segment_windows(dataset_dir: Path) -> dict:
    """{segment name: (first timestamp, last timestamp)}."""
    out = {}
    for f in sorted((dataset_dir / 'samples').glob('segment_*.csv')):
        t = pd.read_csv(f, usecols=['TIMESTAMP'], parse_dates=['TIMESTAMP'],
                        encoding='utf-8', encoding_errors='replace')['TIMESTAMP']
        out[f.name] = (t.iloc[0], t.iloc[-1])
    return out


def segment_valid(dataset_dir: Path, segment: str, columns: list) -> bool:
    t = pd.read_csv(dataset_dir / 'samples' / segment,
                    encoding='utf-8', encoding_errors='replace')
    use = [c for c in columns if c in t.columns]
    return bool(use) and not t[use].isna().values.any()


def replicate_names(dataset_dir: Path, segment: str, subdir: str) -> list:
    """The file names for *segment* under the sample subdirectory a run trains on."""
    if subdir != 'mc_replicates':
        return [segment]
    stem = segment[:-len('.csv')]
    return sorted(p.name for p in (dataset_dir / subdir).glob(stem + '_mc_*.csv'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--old-consolidated', type=Path, required=True,
                    help='The consolidated CSV the existing runs were produced from, '
                         'used to find which scored segments changed.')
    ap.add_argument('--new-consolidated', type=Path,
                    default=REPO_ROOT / 'data' / 'output' / 'regression' / 'Consolidated_sparse.csv')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report what would change and write the plan, evaluate nothing.')
    ap.add_argument('--include-search-runs', action='store_true',
                    help='Also re-evaluate configurations the search tried but did not '
                         'retain. They are never scored by z8, so this only costs time.')
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit('Root not found: %s' % root)

    print('[INFO] Finding predictor values that changed ...')
    changed = changed_cells(args.old_consolidated, args.new_consolidated)
    print('[INFO] %d predictor column(s) changed: %s'
          % (len(changed), ', '.join(sorted(changed)) or 'none'))

    plan_rows = []
    for dataset_dir in sorted(root.glob('MC_*')):
        sweeps = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweeps.is_dir():
            continue
        windows = segment_windows(dataset_dir)
        all_segments = set(windows)
        valid_cache: dict = {}
        retained = z8._retained_keys(sweeps)

        for run in sorted(p for p in sweeps.iterdir() if p.is_dir()):
            train_f, test_f = run / 'train_files.txt', run / 'test_files.txt'
            cfgs = sorted(run.glob('config_evaluate_*.yml'))
            if not (train_f.exists() and test_f.exists() and cfgs):
                continue
            if not args.include_search_runs:
                # Same membership test z8 applies, by feature tag or subset label.
                family = z8._family_of(run.name)
                tag = re.search(r'(f\d+_[0-9a-f]+)', run.name)
                sub = re.search(r'_((?:shap_)?[klms]\d+)$', run.name)
                keys = {(family, tag.group(1)) if tag else None,
                        (family, sub.group(1)) if sub else None}
                if family is None or not (keys & retained):
                    continue
            cfg = yaml.safe_load(open(cfgs[0], encoding='utf-8')) or {}
            data_cfg = cfg.get('data', {}) or {}
            columns = list(data_cfg.get('input_columns', []))
            subdir = str(data_cfg.get('sample_subdir', 'samples'))
            if not columns:
                continue

            # Plan from the original split whenever a previous pass already extended
            # this run. Reading the extended file instead would show the recovered
            # segments as already present, drop the run from the plan, and silently
            # skip it -- so an interrupted pass would leave runs extended but never
            # re-evaluated. Planning from the backup makes the pass resumable.
            backup = run / 'test_files.pre_gapfill.txt'
            source_test = backup if backup.exists() else test_f

            train_raw = [x for x in train_f.read_text(encoding='utf-8').split() if x.strip()]
            test_raw = [x for x in source_test.read_text(encoding='utf-8').split() if x.strip()]
            train_base = {base_name(x) for x in train_raw}
            test_base = {base_name(x) for x in test_raw}
            if not train_base or not test_base:
                continue
            boundary = max(seg_index(x) for x in train_base)

            # (1) recovered segments, admissible only after the boundary
            never_split = all_segments - train_base - test_base
            recovered, before_boundary = [], []
            for s in sorted(never_split):
                key = (s, tuple(columns))
                if key not in valid_cache:
                    valid_cache[key] = segment_valid(dataset_dir, s, columns)
                if not valid_cache[key]:
                    continue
                (recovered if seg_index(s) > boundary else before_boundary).append(s)

            # (2) already-scored segments whose inputs moved
            touched = []
            for c in columns:
                for ts in changed.get(c, ()):
                    for s in test_base:
                        lo, hi = windows[s]
                        if lo <= ts <= hi:
                            touched.append(s)
            touched = sorted(set(touched))

            if not recovered and not touched:
                continue
            plan_rows.append({
                'dataset': dataset_dir.name,
                'run': run.name,
                'config': str(cfgs[0]),
                'sample_subdir': subdir,
                'n_test_before': len(test_base),
                'recovered_after_boundary': ';'.join(recovered),
                'n_recovered': len(recovered),
                'recovered_before_boundary_skipped': len(before_boundary),
                'n_inputs_changed': len(touched),
                'reason': ('extend+rescore' if recovered and touched
                           else 'extend' if recovered else 'rescore'),
            })

    plan = pd.DataFrame(plan_rows)
    out_dir = root / 'summaries'
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / 'reevaluation_plan.csv'
    plan.to_csv(plan_path, index=False)
    print('[INFO] Wrote %s' % plan_path)

    if plan.empty:
        print('[INFO] Nothing to do.')
        return 0

    print()
    print('runs needing work: %d' % len(plan))
    print(plan.groupby('reason').size().to_string())
    print()
    print('per target:')
    g = plan.groupby('dataset').agg(runs=('run', 'size'),
                                    recovered=('n_recovered', 'sum'),
                                    skipped_pre_boundary=('recovered_before_boundary_skipped', 'sum'))
    print(g.to_string())

    if args.dry_run:
        print('\n[INFO] --dry-run: no split extended, nothing evaluated.')
        return 0

    print()
    failures = []
    for i, r in plan.iterrows():
        run_dir = root / r['dataset'] / 'forecasts' / 'feature_sweeps' / r['run']
        dataset_dir = root / r['dataset']
        test_f = run_dir / 'test_files.txt'

        if r['n_recovered']:
            backup = run_dir / 'test_files.pre_gapfill.txt'
            if not backup.exists():
                backup.write_text(test_f.read_text(encoding='utf-8'), encoding='utf-8')
            # Rebuild from the original list rather than appending to whatever is
            # there, so re-running cannot compound an earlier partial extension.
            names = [x for x in backup.read_text(encoding='utf-8').split() if x.strip()]
            for s in r['recovered_after_boundary'].split(';'):
                names.extend(replicate_names(dataset_dir, s, r['sample_subdir']))
            names = sorted(set(names), key=lambda n: (seg_index(n), n))
            test_f.write_text('\n'.join(names) + '\n', encoding='utf-8')

        print('  [EVAL] %-34s %-42s %s' % (r['dataset'][:34], r['run'][:42], r['reason']))
        try:
            subprocess.run([sys.executable, 'src/f_Evaluate.py', '--config', r['config']],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            failures.append((r['dataset'], r['run'], exc.stderr.decode(errors='replace')[-500:]))
            print('     [FAIL] %s' % failures[-1][2].strip().splitlines()[-1:])

    print()
    if failures:
        print('[ERROR] %d run(s) failed to re-evaluate:' % len(failures))
        for ds, run, err in failures[:10]:
            print('   %s / %s' % (ds, run))
        return 1
    print('[INFO] Re-evaluated %d run(s) with no failures.' % len(plan))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
