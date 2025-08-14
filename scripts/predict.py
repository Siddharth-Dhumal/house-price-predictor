import argparse
import json
import os
from datetime import datetime
import joblib
import pandas as pd
from src.config import load_config

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to YAML config")
	parser.add_argument("--json", required=True, help="Path to a JSON file with one example")
	return parser.parse_args()

def load_example(json_path: str) -> dict:
	with open(json_path, "r") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError("Input JSON must contain one object")
	return data

def validate_and_frame(ex: dict, numerical: list[str], categorical: list[str]) -> pd.DataFrame:
	missing = [c for c in (numerical + categorical) if c not in ex]
	if missing:
		raise ValueError(f"Missing keys in input JSON: {missing}")

	for c in numerical:
		if not isinstance(ex[c], (int, float)):
			raise TypeError(f"Field '{c}' must be a number; got {type(ex[c]).__name__}")

	for c in categorical:
		if not isinstance(ex[c], str):
			raise TypeError(f"Field '{c}' must be a string; got {type(ex[c]).__name__}")

	row = {c: ex[c] for c in (numerical + categorical)}
	df = pd.DataFrame([row])
	return df

def main():
	args = parse_args()
	cfg = load_config(args.config)

	if not os.path.exists(cfg.model_path):
		raise FileNotFoundError(f"Model not found at {cfg.model_path}. Run the training first")

	pipe = joblib.load(cfg.model_path)

	ex = load_example(args.json)
	X = validate_and_frame(ex, cfg.numerical, cfg.categorical)

	yhat = pipe.predict(X)
	pred = float(yhat[0])
	print(f"Predicted median house value: {pred:,.2f}")

	os.makedirs(cfg.reports_dir, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = os.path.join(cfg.reports_dir, f"prediction_{ts}.csv")

	# Include inputs + prediction in one row
	out_row = {
	    **{k: X.iloc[0][k] for k in (cfg.numerical + cfg.categorical)},
	    "predicted_median_house_value": pred
	}
	pd.DataFrame([out_row]).to_csv(out_path, index=False)
	print(f"Saved prediction report to: {out_path}")

if __name__ == "__main__":
	main()