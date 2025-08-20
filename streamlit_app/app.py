import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st 
from config import load_config

st.set_page_config(
	page_title="Housing UI",
	layout="wide"
)

st.sidebar.header("Configuration")
default_cfg_path = "configs/default.yaml"

cfg_path = st.sidebar.text_input("Config file path", value=default_cfg_path)

def load_cfg_cached(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found at: {path}")
	return load_config(path)

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