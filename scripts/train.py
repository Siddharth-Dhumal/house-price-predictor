import argparse
import os
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.data_loader import DatasetLoader
from src.pipeline import build_preprocessor, build_model_pipeline

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to YAML config")
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

	pre = build_preprocessor(cfg.numerical, cfg.categorical)
	pipe = build_model_pipeline(pre)

	pipe.fit(X_train, y_train)

	os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
	joblib.dump(pipe, cfg.model_path)

	print("Training completed. Saved pipeline to:", cfg.model_path)

if __name__ == "__main__":
	main()