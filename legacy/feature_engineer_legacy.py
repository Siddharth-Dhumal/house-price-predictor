import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

class FeatureEngineer:
	def __init__(self):
		self.scaler = StandardScaler()

	def transform(self, df):
		clean_df = df.dropna()

		X = clean_df.drop("median_house_value", axis = 1)
		X = pd.get_dummies(X)

		self.feature_names = X.columns.tolist()

		y = clean_df["median_house_value"]

		scaled_X = self.scaler.fit_transform(X)
		dump(self.scaler, "models/scaler.pkl")

		X_train, X_test, y_train, y_test = train_test_split(
			scaled_X, y, random_state = 42, test_size = 0.2, shuffle = True)

		return X_train, X_test, y_train, y_test, self.feature_names