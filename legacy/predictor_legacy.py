from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Predictor:
	def __init__(self, model_path, feature_names):
		self.model = load(model_path)
		self.feature_names = feature_names
		self.scaler = load("models/scaler.pkl")

	def predict(self, new_data):
		input_df = pd.DataFrame([new_data])
		input_df = pd.get_dummies(input_df)

		for col in self.feature_names:
			if col not in input_df.columns:
				input_df[col] = 0

		input_df = input_df[self.feature_names]

		input_df_scaled = self.scaler.transform(input_df)

		price_prediction = self.model.predict(input_df_scaled)

		return price_prediction[0]