# Consolidated Training Script (e_Train.py)

## Overview

The `e_Train.py` script consolidates the functionality of three separate training scripts (`e1_TrainTransformer.py`, `e2_TrainXGBRegressor.py`, and `e3_TrainXGBClassifier.py`) into a single configurable script that supports training Transformer, XGBoost Regressor, and XGBoost Classifier models.

The script uses YAML or JSON configuration files to manage all hyperparameters, data paths, and model-specific settings, eliminating the need to edit Python code for different experiments.

## Key Features

- **Single Unified Interface**: Train all three model types with the same script
- **Configuration-Driven**: All parameters managed via YAML/JSON config files
- **Sensible Defaults**: Built-in defaults for all hyperparameters with console notifications when defaults are applied
- **Flexible Configuration**: Mix and match hyperparameters—only specify what you need to override
- **Separate Hyperparameter Sections**: Each model type has its own hyperparameter dictionary for clarity
- **Configurable Model Names**: The `model_name` field allows custom naming of trained models

## Usage

### Basic Syntax

```bash
python e_Train.py --config <path_to_config_file>
```

### Example Runs

```bash
# Train Transformer
python e_Train.py --config ../config_transformer_example.yaml

# Train XGBoost Regressor
python e_Train.py --config ../config_xgb_regressor_example.yaml

# Train XGBoost Classifier
python e_Train.py --config ../config_xgb_classifier_example.yaml
```

## Configuration File Structure

Configuration files use YAML or JSON format. Below is the general structure:

### Required Fields

```yaml
model_type: transformer  # or 'xgb_regressor' or 'xgb_classifier'
model_name: my_custom_model_name

data:
  data_dir: "../data/output/regression/Farge"
  forecast_name: "nowcast"
  input_columns:
    - 'Column 1'
    - 'Column 2'
    # ... more columns
  output_columns:
    - 'Target Column'
  input_row_1: 0
  input_row_2: 166
  output_rows: -1
```

### Optional Fields (Applied with Defaults if Omitted)

```yaml
# Data split configuration (uses defaults if omitted)
data_split:
  random_state: 35
  test_size: 0.2
  reuse_split: false
  split_source: null
  split_type: random

# Device configuration (auto-detects CUDA if omitted)
device: cuda

# Model hyperparameters (uses model-specific defaults if omitted)
hyperparameters:
  # Transformer-specific hyperparameters
  model_dim: 128
  num_heads: 4
  num_layers: 8
  dropout: 0.1
  batch_size: 10
  num_epochs: 100
  learning_rate: 0.0001
  # ... etc
```

## Configuration File Examples

Three example configuration files are provided in the root directory:

1. **config_transformer_example.yaml** - Transformer model configuration
2. **config_xgb_regressor_example.yaml** - XGBoost Regressor configuration
3. **config_xgb_classifier_example.yaml** - XGBoost Classifier configuration

Copy and modify these examples as needed for your experiments.

## Default Hyperparameters

### Transformer Defaults
| Parameter | Default |
|-----------|---------|
| model_dim | 128 |
| num_heads | 4 |
| num_layers | 8 |
| dropout | 0.1 |
| batch_size | 10 |
| num_epochs | 100 |
| loss_threshold | 0.000001 |
| learning_rate | 1e-4 |
| patience | 10 |

### XGBoost Regressor Defaults
| Parameter | Default |
|-----------|---------|
| metric | rmse |
| tree_method | hist |
| objective | reg:squarederror |
| n_estimators | 1100 |
| max_depth | 10 |
| subsample | 0.2 |
| colsample_bytree | 0.8 |
| learning_rate | 0.01 |
| n_jobs | -1 |
| early_stopping_rounds | 200 |

### XGBoost Classifier Defaults
| Parameter | Default |
|-----------|---------|
| eval_metric | logloss |
| tree_method | hist |
| objective | binary:logistic |
| n_estimators | 1500 |
| max_depth | 10 |
| subsample | 0.2 |
| colsample_bytree | 0.8 |
| learning_rate | 0.01 |
| n_jobs | -1 |
| early_stopping_rounds | 50 |

### Data Split Defaults
| Parameter | Default |
|-----------|---------|
| random_state | 42 |
| test_size | 0.2 |
| reuse_split | false |
| split_source | null |
| split_type | random |

## Default Application Behavior

When hyperparameters are **not specified** in your config file, the script will:

1. **Apply the default value** for that model type
2. **Print a console message** indicating which default was applied:
   ```
   [DEFAULT] hyperparameters.learning_rate = 0.01
   ```

This allows you to start with minimal configuration and only override what you need.

## Output Structure

All models save outputs in the following directory structure:

```
<data_dir>/
  forecasts/
    <forecast_name>/
      <model_name>/
        xgboost_model.json  (for XGBoost models)
        transformer_model.pt  (for Transformer model)
        loss_plot.png
        config.json  (model configuration)
```

## Console Output

The script provides detailed console output showing:
- Configuration loading and defaults applied
- Device used (CPU/CUDA)
- Data split information
- Training progress
- Model save locations
- Loss plots

Example:
```
================================================================================
LOADING CONFIGURATION
================================================================================
Config file: ../config_xgb_regressor_example.yaml
Model type: xgb_regressor

Applying defaults:
  [DEFAULT] hyperparameters.metric = rmse
  [DEFAULT] device = cuda

Using device: cuda

================================================================================
LOADING AND SPLITTING DATA
================================================================================
Training samples: 800
Test samples: 200

================================================================================
TRAINING XGBOOST REGRESSOR MODEL
================================================================================
Training XGBRegressor...
[0]	validation_0-rmse:1.23456	validation_1-rmse:1.45678
...
```

## Minimal Configuration Example

The simplest possible config file (using all defaults except required fields):

```yaml
model_type: xgb_regressor
model_name: my_model

data:
  data_dir: "../data/output/regression/Farge"
  forecast_name: "nowcast"
  input_columns:
    - 'Pfl - Water temperature (°C)'
    - 'Pfl - Sp Cond (microS_cm)'
  output_columns:
    - '01-Farge_res'
  input_row_1: 0
  input_row_2: 166
  output_rows: -1
```

All other settings will use sensible defaults automatically.

## Troubleshooting

### Configuration File Not Found
```
FileNotFoundError: Configuration file not found: <path>
```
**Solution**: Check the config file path is correct and file exists.

### Missing Required Configuration Field
```
ValueError: Missing required config field: model_type
```
**Solution**: Ensure your config file includes `model_type`, `model_name`, and `data` sections.

### Unknown Model Type
```
ValueError: Unknown model_type: <type>
```
**Solution**: Use one of: `transformer`, `xgb_regressor`, `xgb_classifier`

### CUDA Not Available
The script automatically falls back to CPU if CUDA is unavailable. You can also explicitly set `device: cpu` in your config.

## Migration from Old Scripts

If you have existing training runs using the old separate scripts, you can migrate them:

1. Note the hyperparameters from the old script
2. Create a new config file with those hyperparameters
3. Run the new consolidated script with the config file

The outputs and model behavior will be identical.
