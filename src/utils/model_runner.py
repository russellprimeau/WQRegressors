import yaml
import copy
import torch
import pandas as pd
from pathlib import Path
from . import training
from .. import f_Evaluate as eval_module

class ModelRunner:
    def __init__(self, base_config_path, tmp_dir):
        self.base_config_path = Path(base_config_path)
        self.tmp_dir = Path(tmp_dir)
        self.base_config = training.load_config(str(self.base_config_path))

    def prepare_variant_config(self, row_count, features, feature_tag):
        cfg_copy = copy.deepcopy(self.base_config)
        if "data" not in cfg_copy:
            raise ValueError(f"Missing data section in {self.base_config_path}")
        data_cfg = cfg_copy["data"]
        source_config_dir = Path(self.base_config.get("__config_dir", self.base_config_path.parent))
        required_data = ["input_row_1", "input_row_2", "forecast_name", "data_dir"]
        for field in required_data:
            if field not in data_cfg:
                raise ValueError(f"Missing data.{field} in {self.base_config_path}")
        base_stop = int(data_cfg["input_row_2"])
        base_start = int(data_cfg["input_row_1"])
        base_span = base_stop - base_start
        if base_span <= 0:
            raise ValueError(f"Invalid input row span in {self.base_config_path}: {base_start}..{base_stop}")
        if row_count > base_span:
            raise ValueError(f"row_count={row_count} exceeds base span={base_span} for {self.base_config_path.name}")
        data_cfg["input_columns"] = list(features)
        data_cfg["input_row_1"] = int(base_stop - row_count)
        data_cfg["input_row_2"] = int(base_stop)
        resolved_data_dir = training._resolve_path_from_config(data_cfg["data_dir"], source_config_dir)
        data_cfg["data_dir"] = str(resolved_data_dir)
        data_cfg["forecast_name"] = f"feature_sweeps/{data_cfg['forecast_name']}_r{row_count:03d}_{feature_tag}"
        forecast_dir = Path(resolved_data_dir) / "forecasts" / "feature_sweeps" / data_cfg["forecast_name"]
        data_cfg["forecast_dir"] = str(forecast_dir.resolve())
        if "evaluation" not in cfg_copy:
            cfg_copy["evaluation"] = {}
        cfg_copy["evaluation"]["run_baselines"] = True
        cfg_copy.pop("__config_dir", None)
        variant_name = f"{self.base_config_path.stem}_r{row_count:03d}_{feature_tag}{self.base_config_path.suffix}"
        variant_path = self.tmp_dir / variant_name
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with open(variant_path, "w", encoding="utf-8") as f:
            if variant_path.suffix.lower() in {".yml", ".yaml"}:
                yaml.safe_dump(cfg_copy, f, sort_keys=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config suffix for variant writing: {variant_path.suffix}")
        return variant_path

    def train(self, config_path, dataset_dir, disable_training_plots=False, disable_eval_plots=False, suppress_training_logs=False):
        config = training.load_config(str(config_path))
        required_fields = ["model_type", "model_name", "data"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field '{field}' in {config_path}")
        model_type = config["model_type"]
        config = training.merge_with_defaults(config, model_type)
        if disable_training_plots:
            config["save_training_plots"] = False
        device = torch.device(config["device"])
        import matplotlib
        matplotlib.use(config["matplotlib_backend"])
        if suppress_training_logs:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train_samples, test_samples, _ = training.load_and_split_data(config)
        else:
            train_samples, test_samples, _ = training.load_and_split_data(config)
        def _run_train():
            if model_type == "transformer":
                training.train_transformer_model(config, train_samples, test_samples)
            elif model_type == "gp_regressor":
                training.train_gp_regressor_model(config, train_samples, test_samples)
            elif model_type == "xgb_regressor":
                training.train_xgb_regressor_model(config, train_samples, test_samples)
            elif model_type == "xgb_classifier":
                training.train_xgb_classifier_model(config, train_samples, test_samples)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        if suppress_training_logs:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _run_train()
        else:
            _run_train()
        data_cfg = config["data"]
        forecast_name = data_cfg["forecast_name"]
        forecast_file_name = Path(str(forecast_name)).name
        forecast_dir = dataset_dir / "forecasts" / "feature_sweeps" / Path(forecast_name)
        return (forecast_dir / f"config_evaluate_{forecast_file_name}.yml").resolve()

    def evaluate(self, eval_config_path, save_plots=True):
        # Always force run_baselines True for evaluation summary output
        with open(eval_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "evaluation" not in cfg:
            cfg["evaluation"] = {}
        cfg["evaluation"]["run_baselines"] = True
        with open(eval_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        eval_result = eval_module.evaluate_single_config(
            str(eval_config_path),
            save_plots_override=save_plots,
        )
        return eval_result

    def extract_results(self, eval_result):
        found_baseline = False
        found_model = False
        for row in eval_result.get("summary_rows", []):
            kind = str(row.get("kind", "")).lower()
            if kind == "model":
                found_model = True
            if kind in ("naive", "seasonal", "linear"):
                found_baseline = True
        for row in eval_result.get("summary_rows", []):
            if str(row.get("kind", "")).lower() == "model":
                return row
        return {}
