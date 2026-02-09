# Parameter Sweeper Guide (g_ParameterSweeper.py)

## Overview

The `g_ParameterSweeper.py` script automates hyperparameter optimization by:

1. **Iterating through parameter combinations** - Sweeps through a defined parameter space
2. **Training models automatically** - Uses `e_Train.py` to train a model for each parameter combination
3. **Saving configurations** - Stores each configuration in a dedicated directory for reproducibility
4. **Evaluating all models** - Compares predictions against baseline models (Naive, Linear, Seasonal)
5. **Visualizing results** - Generates comprehensive evaluation plots and metrics summaries

## Key Features

- **Automatic Config Management**: Creates, modifies, and saves YAML config files for each iteration
- **Subprocess Training**: Uses subprocess to invoke `e_Train.py`, isolating each training run
- **Organized Output**: Results stored in `sweep_results/iteration_NNN/` directories
- **Reproducibility**: Each iteration's exact configuration saved for future reference
- **Comparison Framework**: All swept models compared against baseline methods
- **Summary Statistics**: Exports performance metrics to CSV

## Usage

### Basic Setup

Edit the main `if __name__ == '__main__':` section in `g_ParameterSweeper.py`:

```python
# Path to base configuration file (will be modified for each sweep iteration)
base_config_path = "../config_xgb_regressor_example.yaml"

# Data directory and forecast name (should match base config)
data_dir = "../data/output/regression/Farge"
forecast_name = "nowcast"
base_model_name = "xgbregressor"

# Define parameter space to sweep
# Format: {parameter_path: [list of values to test]}
# parameter_path uses dot notation: "hyperparameters.learning_rate"
sweep_parameters = {
    "hyperparameters.learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05],
    # "hyperparameters.max_depth": [5, 10, 15, 20],
    # "hyperparameters.n_estimators": [500, 1000, 1500],
}
```

### Run the Sweep

```bash
cd src
python g_ParameterSweeper.py
```

## Parameter Specification

Parameters are specified using **dot notation** to navigate the config structure:

```python
sweep_parameters = {
    # Single parameter sweep
    "hyperparameters.learning_rate": [0.001, 0.01, 0.1],
    
    # Multiple parameters (creates all combinations)
    "hyperparameters.max_depth": [5, 10, 15],
    "hyperparameters.n_estimators": [500, 1000],
    
    # Data parameters
    "data.input_row_2": [50, 100, 150],
}
```

### Multi-Parameter Sweeps

When specifying multiple parameters, the script generates **all combinations**:

```python
sweep_parameters = {
    "hyperparameters.learning_rate": [0.01, 0.02],      # 2 values
    "hyperparameters.max_depth": [5, 10, 15],           # 3 values
}
# Total iterations: 2 × 3 = 6 combinations
```

## Output Structure

```
data_dir/
  sweep_results/
    iteration_000/
      config.yaml              # Config used for this iteration
      ../                      # Training output (models, plots, etc.)
    iteration_001/
      config.yaml
      ../
    iteration_002/
      config.yaml
      ../
    sweep_summary.csv          # Summary metrics for all iterations + baselines
```

### Config File Contents

Each `config.yaml` documents exactly which hyperparameters were used:

```yaml
model_type: xgb_regressor
model_name: xgbregressor_iter000

data:
  data_dir: "../data/output/regression/Farge"
  forecast_name: "nowcast"
  # ... other data config

hyperparameters:
  metric: rmse
  learning_rate: 0.001    # <- Modified for this iteration
  max_depth: 10
  # ... other hyperparameters
```

### Sweep Summary CSV

The `sweep_summary.csv` file compares all models:

```
Model,MAE,RMSE,R2
Iter 0 (learning_rate=0.001),0.1234,0.1567,0.8901
Iter 1 (learning_rate=0.005),0.1123,0.1456,0.8956
Iter 2 (learning_rate=0.01),0.1089,0.1401,0.9012
Naive,0.2345,0.2876,0.6543
Linear,0.1567,0.1923,0.8234
Seasonal,0.1456,0.1812,0.8456
```

## Console Output Example

```
================================================================================
LOADING CONFIGURATION
================================================================================
Config file: ../config_xgb_regressor_example.yaml
Base config loaded: ../config_xgb_regressor_example.yaml

Sweep results directory: ../data/output/regression/Farge/sweep_results

================================================================================
PARAMETER SWEEP LOOP
================================================================================

================================================================================
SWEEP ITERATION 1
================================================================================
  hyperparameters.learning_rate = 0.001
  Config saved to: ../data/output/regression/Farge/sweep_results/iteration_000/config.yaml
  
  Training model...
  [Training output from e_Train.py...]
  Model loaded successfully

================================================================================
SWEEP ITERATION 2
================================================================================
  hyperparameters.learning_rate = 0.005
  [...]

================================================================================
COMPLETED 5 SUCCESSFUL ITERATIONS
================================================================================

================================================================================
EVALUATING SWEPT MODELS
================================================================================

Evaluating iteration 0: xgbregressor_iter000
  Evaluation successful

[...]

Baseline models evaluated successfully

Generating visualizations...

Sweep summary saved to: ../data/output/regression/Farge/sweep_results/sweep_summary.csv
Model,MAE,RMSE,R2
Iter 0 (learning_rate=0.001),0.1234,0.1567,0.8901
[...]

================================================================================
SWEEP COMPLETE
================================================================================
```

## Common Parameter Ranges

### XGBoost Regressor

```python
sweep_parameters = {
    "hyperparameters.learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05],
    "hyperparameters.max_depth": [5, 8, 10, 15],
    "hyperparameters.n_estimators": [500, 1000, 1500],
    "hyperparameters.subsample": [0.5, 0.7, 0.9],
}
```

### XGBoost Classifier

```python
sweep_parameters = {
    "hyperparameters.learning_rate": [0.001, 0.005, 0.01, 0.02],
    "hyperparameters.max_depth": [5, 10, 15, 20],
    "hyperparameters.n_estimators": [500, 1000, 1500, 2000],
}
```

### Transformer

```python
sweep_parameters = {
    "hyperparameters.learning_rate": [1e-5, 1e-4, 1e-3],
    "hyperparameters.model_dim": [64, 128, 256],
    "hyperparameters.num_layers": [4, 6, 8],
    "hyperparameters.dropout": [0.05, 0.1, 0.2],
}
```

### Data-Level Parameters

```python
sweep_parameters = {
    "data.input_row_2": [50, 100, 150, 200],  # Sequence length
    "data_split.test_size": [0.1, 0.2, 0.3],
}
```

## Performance Tips

1. **Start with fewer values**: Test 2-3 values per parameter before expanding
2. **Use coarse-to-fine approach**: Do broad sweep, then narrow down promising regions
3. **Parallelize if possible**: Modify script to run multiple training jobs simultaneously (requires careful disk management)
4. **Monitor disk space**: Each model takes ~10-100 MB. Plan accordingly for large sweeps

## Troubleshooting

### Training failures in some iterations

Check the iteration-specific output directory:
- Model training output: `sweep_results/iteration_NNN/` (created by e_Train.py)
- Config file: `sweep_results/iteration_NNN/config.yaml`

The script will skip failed iterations and continue with the rest.

### Insufficient disk space

Compress old sweep results:
```bash
cd data/output/regression/Farge
tar -czf sweep_results_backup.tar.gz sweep_results/iteration_000 sweep_results/iteration_001
```

### Memory errors during training

Reduce batch size or model dimensions in your config:
```python
sweep_parameters = {
    "hyperparameters.batch_size": [5, 10, 20],
}
```

### Config not modifying parameters correctly

Verify dot notation matches your config structure. Print the config to check:
```yaml
# In your base config:
hyperparameters:
  learning_rate: 0.01  # Use: "hyperparameters.learning_rate"
  
data:
  input_row_2: 100     # Use: "data.input_row_2"
```

## Advanced: Extending the Script

### Adding Custom Evaluation Metrics

Edit the visualization/metrics section to add custom metrics:

```python
# In the metrics collection loop:
summary_data.append({
    "Model": all_labels[i],
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Custom_Metric": custom_metric_value,  # Add here
})
```

### Filtering Iterations

Skip certain iterations by adding logic:

```python
for iteration, values in enumerate(product(*param_values)):
    # Skip every other iteration
    if iteration % 2 == 1:
        continue
```

### Saving Prediction Files

Store detailed predictions for post-processing:

```python
for config_entry in sweep_configs:
    iteration = config_entry["iteration"]
    # After evaluation:
    np.save(f"sweep_results/iteration_{iteration:03d}/predictions.npy", preds)
    np.save(f"sweep_results/iteration_{iteration:03d}/targets.npy", targets)
```

## Integration with e_Train.py

The parameter sweeper automatically:
1. Creates YAML configs for each iteration
2. Calls: `python e_Train.py --config sweep_results/iteration_NNN/config.yaml`
3. Waits for training to complete
4. Loads the trained model from `forecasts/<forecast_name>/<model_name>/`
5. Evaluates predictions against test data

Ensure your `e_Train.py` is in the same directory as `g_ParameterSweeper.py` for subprocess calls to work correctly.
