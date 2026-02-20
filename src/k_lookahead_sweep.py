"""
Lookahead sweep script: For each dataset, take the best-performing model/config (from feature_sweeps),
then iteratively remove rows from the bottom of the predictor set (lookahead = 1,2,6,12,24,48,96,120,167),
retrain and evaluate the model, and save metrics/plots to forecasts/lookahead_sweeps/.
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import subprocess

LOOKAHEADS = [1,2,6,12,24,48,96,120,167]

# Helper: Find best config for each dataset from feature_sweeps
def find_best_configs(data_root, dataset_prefix):
    best_configs = []
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(dataset_prefix):
            continue
        sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweep_dir.exists():
            continue
        # Use the only feature_selected_subsets_r*.csv file
        selected_files = sorted(sweep_dir.glob('feature_selected_subsets_r*.csv'))
        if not selected_files:
            continue
        selected = pd.read_csv(selected_files[0])
        if selected.empty:
            continue
        # Find the row with the highest r2 value
        if 'r2' in selected.columns:
            best_idx = selected['r2'].idxmax()
            best = selected.iloc[best_idx]
        else:
            best = selected.iloc[0]
        # Find config file for this subset
        configs_dir = sweep_dir / 'configs'
        config_name = f"config_evaluate_{best['feature_tag']}_r{int(best['row_count']):03d}.yml"
        config_path = None
        for cfg in configs_dir.glob(f"*r{int(best['row_count']):03d}_{best['feature_tag']}*.yml"):
            config_path = cfg
            break
        if config_path is None:
            continue
        best_configs.append((dataset_dir, config_path, best))
    return best_configs

# Helper: Modify config for lookahead (remove rows from predictors)
def modify_config_for_lookahead(config_path, lookahead, output_dir):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    base_start = int(data_cfg['input_row_1'])
    base_stop = int(data_cfg['input_row_2'])
    new_stop = base_stop - lookahead
    if new_stop <= base_start:
        return None  # Not enough rows
    data_cfg['input_row_2'] = new_stop
    # Update forecast_name and output dir
    tag = f"lookahead_{lookahead:03d}"
    data_cfg['forecast_name'] = f"lookahead_sweeps/{data_cfg['forecast_name']}_{tag}"
    out_cfg = output_dir / f"config_lookahead_{lookahead:03d}.yml"
    with open(out_cfg, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_cfg

# Main sweep logic
def run_lookahead_sweep(data_root, dataset_prefix):
    best_configs = find_best_configs(data_root, dataset_prefix)
    for dataset_dir, base_config, best_row in best_configs:
        print(f"\n[DATASET] {dataset_dir.name}")
        sweep_dir = dataset_dir / 'forecasts' / 'lookahead_sweeps'
        sweep_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        for lookahead in LOOKAHEADS:
            mod_cfg = modify_config_for_lookahead(base_config, lookahead, sweep_dir)
            if mod_cfg is None:
                print(f"  [SKIP] Lookahead {lookahead}: not enough rows.")
                continue
            # Retrain and evaluate
            cmd = [
                'python', 'src/f_Evaluate.py',
                '--config', str(mod_cfg),
                '--no-plots'  # We'll plot after collecting all metrics
            ]
            print(f"  [RUN] Lookahead {lookahead}: {mod_cfg.name}")
            subprocess.run(cmd, check=True)
            # Collect metrics from evaluation_summary.csv
            # Read config to get forecast_name for this lookahead
            with open(mod_cfg, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg['data']
            eval_dir = dataset_dir / 'forecasts' / Path(data_cfg['forecast_name'])
            eval_csv = eval_dir / 'evaluation_summary.csv'
            if eval_csv.exists():
                df = pd.read_csv(eval_csv)
                df['lookahead'] = lookahead
                metrics.append(df)
        # Save combined metrics
        if metrics:
            all_metrics = pd.concat(metrics, ignore_index=True)
            all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)
            # Plot RMSE and R2 vs lookahead
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            plt.plot(all_metrics['lookahead'], all_metrics['rmse'], marker='o', label='RMSE')
            plt.xlabel('Lookahead (hours)')
            plt.ylabel('RMSE')
            plt.title(f'{dataset_dir.name} - RMSE vs Lookahead')
            plt.grid(True)
            plt.savefig(sweep_dir / 'rmse_vs_lookahead.png')
            plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(all_metrics['lookahead'], all_metrics['r2'], marker='o', label='R2')
            plt.xlabel('Lookahead (hours)')
            plt.ylabel('R2')
            plt.title(f'{dataset_dir.name} - R2 vs Lookahead')
            plt.grid(True)
            plt.savefig(sweep_dir / 'r2_vs_lookahead.png')
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lookahead sweep for best models in each dataset')
    parser.add_argument('--data-root', type=str, default='data/output/regression')
    parser.add_argument('--dataset-prefix', type=str, default='MC')
    args = parser.parse_args()
    run_lookahead_sweep(args.data_root, args.dataset_prefix)
