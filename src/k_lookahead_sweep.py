"""
Lookahead sweep script: For each dataset, take the best-performing model/config (from feature_sweeps),
then iteratively remove rows from the bottom of the predictor set (lookahead = 1,2,6,12,24,48,96,120,167),
retrain and evaluate the model, and save metrics/plots to forecasts/lookahead_sweeps/.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import subprocess

LOOKAHEADS = [0,1,2,6,12,24,48,96,120,167]

# Helper: Find best config for each dataset from feature_sweeps
def find_best_configs(data_root, dataset_prefix):
    best_configs = []
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith(dataset_prefix):
            continue
        sweep_dir = dataset_dir / 'forecasts' / 'feature_sweeps'
        if not sweep_dir.exists():
            continue
        metrics_file = sweep_dir / 'feature_sweep_final_metrics.csv'
        if not metrics_file.exists():
            continue
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df.empty or 'r2' not in metrics_df.columns:
            continue
        best_idx = metrics_df['r2'].idxmax()
        best = metrics_df.iloc[best_idx]
        # Find config file for this subset and model
        configs_dir = sweep_dir / 'configs'
        model_map = {
            'Transformer': 'transformer_01',
            'XGBRegressor': 'xgb_01',
            'GPRegressor': 'gp_01',
        }
        model_name = str(best['model'])
        mapped_model_name = model_map.get(model_name, model_name.lower())
        config_path = None
        for cfg in configs_dir.glob(f"*{mapped_model_name}*r{int(best['row_count']):03d}_{best['feature_tag']}*.yml"):
            config_path = cfg
            break
        if config_path is None:
            # fallback: match by feature_tag and row_count only
            for cfg in configs_dir.glob(f"*r{int(best['row_count']):03d}_{best['feature_tag']}*.yml"):
                config_path = cfg
                break
        if config_path is None:
            continue
        # Map model names to directory short names
        model_map = {
            'Transformer': 'transformer_01',
            'XGBRegressor': 'xgb_01',
            'GPRegressor': 'gp_01',
        }
        model_name = str(best['model'])
        model_dir_name = model_map.get(model_name, model_name.lower())
        row_count_val = int(best['row_count'])
        feature_tag = str(best['feature_tag'])
        subset_rank_val = int(best['subset_rank'])
        subset_rank_str = f"k{subset_rank_val:02d}"
        # Directory format: model_01_r168_featuretag_k01
        forecast_dir = dataset_dir / 'forecasts' / 'feature_sweeps' / f"{model_dir_name}_r{row_count_val}_{feature_tag}_{subset_rank_str}"
        best_configs.append((dataset_dir, config_path, best, forecast_dir))
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
    # Set forecast_name to lookahead_sweeps/{lookahead}_hr
    lookahead_dir = f"lookahead_sweeps/{lookahead}_hr"
    data_cfg['forecast_name'] = lookahead_dir
    out_cfg = output_dir / f"{lookahead}_hr" / "config_evaluate.yml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cfg, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_cfg

# Main sweep logic
def run_lookahead_sweep(data_root, dataset_prefix):
    best_configs = find_best_configs(data_root, dataset_prefix)
    for dataset_dir, base_config, best_row, forecast_dir in best_configs:
        print(f"\n[DATASET] {dataset_dir.name}")
        sweep_dir = dataset_dir / 'forecasts' / 'lookahead_sweeps'
        sweep_dir.mkdir(parents=True, exist_ok=True)
        # No longer copy files to sweep_dir; handled per-lookahead below
        if not forecast_dir.exists():
            print(f"  [WARN] Forecast directory not found: {forecast_dir}")
        metrics = []
        for lookahead in LOOKAHEADS:
            lookahead_dir = sweep_dir / f"{lookahead}_hr"
            mod_cfg = modify_config_for_lookahead(base_config, lookahead, sweep_dir)
            if mod_cfg is None:
                print(f"  [SKIP] Lookahead {lookahead}: not enough rows.")
                continue
            # Read config to get forecast_name for this lookahead
            with open(mod_cfg, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg['data']
            eval_dir = dataset_dir / 'forecasts' / Path(data_cfg['forecast_name'])
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Retrain the model for this lookahead config
            # Assumes e_Train.py supports --config argument and writes model files to eval_dir
            train_cmd = [
                sys.executable, 'src/e_Train.py',
                '--config', str(mod_cfg)
            ]
            print(f"  [TRAIN] Lookahead {lookahead}: {mod_cfg.name}")
            # Suppress stdout/stderr for training
            with open(os.devnull, 'w') as devnull:
                subprocess.run(train_cmd, check=True, stdout=devnull, stderr=devnull)

            # Evaluate the newly trained model
            eval_cmd = [
                sys.executable, 'src/f_Evaluate.py',
                '--config', str(mod_cfg),
                '--no-plots'
            ]
            print(f"  [EVAL] Lookahead {lookahead}: {mod_cfg.name}")
            # Suppress stdout/stderr for evaluation
            with open(os.devnull, 'w') as devnull:
                subprocess.run(eval_cmd, check=True, stdout=devnull, stderr=devnull)
            eval_csv = eval_dir / 'evaluation_summary.csv'
            if eval_csv.exists():
                df = pd.read_csv(eval_csv)
                # Add columns for included/excluded input rows
                with open(mod_cfg, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                data_cfg = cfg['data']
                input_row_1 = int(data_cfg['input_row_1'])
                input_row_2 = int(data_cfg['input_row_2'])
                total_rows = input_row_2 - input_row_1
                # Try to get the original max rows from the base config
                with open(base_config, 'r', encoding='utf-8') as f:
                    base_cfg = yaml.safe_load(f)
                base_row_1 = int(base_cfg['data']['input_row_1'])
                base_row_2 = int(base_cfg['data']['input_row_2'])
                base_total_rows = base_row_2 - base_row_1
                rows_excluded = base_total_rows - total_rows
                df['input_rows_included'] = total_rows
                df['input_rows_excluded'] = rows_excluded
                df['lookahead'] = lookahead
                metrics.append(df)
        # Save metrics: only one row per lookahead (test set), lookahead as first column, label omitted
        if metrics:
            import matplotlib.pyplot as plt
            # Only keep rows where kind == 'test' (or label contains 'test')
            filtered = []
            for df in metrics:
                if 'kind' in df.columns:
                    filtered.append(df[df['kind'] == 'test'])
                elif 'label' in df.columns:
                    filtered.append(df[df['label'].str.contains('test', case=False, na=False)])
                else:
                    filtered.append(df)
            all_metrics = pd.concat(filtered, ignore_index=True)
            # Drop label/model columns if present
            drop_cols = [col for col in ['label', 'kind'] if col in all_metrics.columns]
            all_metrics = all_metrics.drop(columns=drop_cols, errors='ignore')
            # Move lookahead to first column if present
            cols = list(all_metrics.columns)
            if 'lookahead' in cols:
                cols.insert(0, cols.pop(cols.index('lookahead')))
                all_metrics = all_metrics[cols]
            all_metrics.to_csv(sweep_dir / 'lookahead_metrics.csv', index=False)
            # Plot RMSE and R2 vs lookahead
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
