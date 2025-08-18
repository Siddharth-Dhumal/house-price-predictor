import argparse
import os
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.data_loader import DatasetLoader
from src.pipeline import build_preprocessor, build_model_pipeline
from src.metrics import evaluate_holdout, baseline_metrics, evaluate_cv

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to YAML config")
	parser.add_argument("--model-preset", default=None, help="Name of model preset from YAML (models.presets.*)")
	return parser.parse_args()

def main():
	args = parse_args()
	cfg = load_config(args.config)

	df = DatasetLoader(cfg.data_path).load_data()
	if df is None or df.empty:
		raise RuntimeError(f"Failed to load data at {cfg.data_path}")

	X = df[cfg.numerical + cfg.categorical]
	y = df[cfg.target]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
	)

	selected = args.model_preset or cfg.models_default
	presets = cfg.models_presets

	print("Selected preset:", selected)
	print("Available presets:", list(presets.keys()))

	if selected not in presets:
		raise KeyError(f"Unknown model preset '{selected}'. Available: {sorted(presets.keys())}")

	preset = presets[selected]
	print("Using model preset:", selected, "->", preset["type"])

	pre = build_preprocessor(cfg.numerical, cfg.categorical)
	pipe = build_model_pipeline(pre, cfg_model=preset)

	cv_stats = evaluate_cv(pipe, X_train, y_train, cv=5)
	print("\nCV (5-fold) on TRAIN:")
	print(f"  RMSE: {cv_stats['cv_rmse_mean']:.2f} ± {cv_stats['cv_rmse_std']:.2f}")
	print(f"  MAE : {cv_stats['cv_mae_mean']:.2f} ± {cv_stats['cv_mae_std']:.2f}")
	print(f"  R²  : {cv_stats['cv_r2_mean']:.3f} ± {cv_stats['cv_r2_std']:.3f}")
	print(f"  MAPE: {cv_stats['cv_mape_mean']:.2f}% ± {cv_stats['cv_mape_std']:.2f}%")

	pipe.fit(X_train, y_train)

	holdout = evaluate_holdout(pipe, X_test, y_test)
	print("\nHoldout TEST:")
	print(f"  RMSE: {holdout['test_rmse']:.2f}")
	print(f"  MAE : {holdout['test_mae']:.2f}")
	print(f"  R²  : {holdout['test_r2']:.3f}")
	print(f"  MAPE: {holdout['test_mape']:.2f}%")

	base = baseline_metrics(y_train, y_test)
	print("\nBaseline (predict train mean) on TEST:")
	print(f"  RMSE: {base['baseline_rmse']:.2f}")
	print(f"  MAE : {base['baseline_mae']:.2f}")
	print(f"  R²  : {base['baseline_r2']:.3f}")
	print(f"  MAPE: {base['baseline_mape']:.2f}%")

	os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
	joblib.dump(pipe, cfg.model_path)

	print("Training completed. Saved pipeline to:", cfg.model_path)

if __name__ == "__main__":
	main()