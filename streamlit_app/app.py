import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st 
from config import load_config
import pandas as pd 
import joblib

st.set_page_config(
	page_title="Housing UI",
	layout="wide"
)

st.sidebar.header("Configuration")
default_cfg_path = "configs/default.yaml"

cfg_path = st.sidebar.text_input("Config file path", value=default_cfg_path)

@st.cache_resource
def load_cfg_cached(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found at: {path}")
	return load_config(path)

@st.cache_resource
def load_model_cached(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model not found at: {path}")
	model = joblib.load(path)
	return model

def fmt_currency(x: float) -> str:
	return str(f"${x:,.2f}")

def build_single_row(inputs: dict, numerical: list[str], categorical: list[str]) -> pd.DataFrame:
	row = {c: inputs[c] for c in (numerical + categorical)}
	df = pd.DataFrame([row])
	return df

cfg = None
error_msg = None

try:
	cfg = load_cfg_cached(cfg_path)
except Exception as e:
	error_msg = str(e)

st.title("California Housing Price Predictor")

if error_msg:
	st.error(f"Could not load config: {error_msg}")
	st.stop()

st.subheader("Configuration summary")

st.markdown(f"Data path: {cfg.data_path}")
st.markdown(f"Model path: {cfg.model_path}")
st.markdown(f"Reports dir: {cfg.reports_dir}")

st.markdown(f"Target: {cfg.target}")
st.markdown(f"Test size: {cfg.test_size} & Random state {cfg.random_state}")

numerical = cfg.numerical
categorical = cfg.categorical

st.markdown("Numerical")
st.write(numerical)

st.markdown("Categorical")
st.write(categorical)

st.success("Config Loaded")

st.divider()
st.subheader("Single Prediction")

model = None 
model_error = None  

try:
	model = load_model_cached(cfg.model_path)
except Exception as e:
	model_error = str(e)

if model_error:
	st.error(f"Could not load model: {model_error}")
	st.stop()

with st.form(key="single_predication"):
	st.caption("Enter all feature values, then select Predict")

	num_values = {}

	for name in cfg.numerical:
		num_values[name] = st.number_input(
			label=f"{name}",
			value=0.0,
			step=0.1
		)

	cat_values = {}

	ocean_options = ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
	cat_values["ocean_proximity"] = st.selectbox("ocean_proximity", options=ocean_options)

	submitted = st.form_submit_button("Predict")

if submitted:
	inputs = {**num_values, **cat_values}
	X = build_single_row(inputs, cfg.numerical, cfg.categorical)

	try:
		yhat = model.predict(X)
		pred = float(yhat[0])
	except Exception as e:
		st.error(f"Prediction failed: {e}")
		st.stop()

	st.success(f"Predicted {cfg.target}: {fmt_currency(pred)}")