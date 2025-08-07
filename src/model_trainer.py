from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
	def __init__(self):
		self.linear_model = LinearRegression()

	def train(self, X_train, y_train):
		self.linear_model.fit(X_train, y_train)

	def evaluate(self, X_test, y_test):
		y_pred = self.linear_model.predict(X_test)

		print("The mean sqaured error is:", mean_squared_error(y_test, y_pred))
		print("The r^2 score is:", r2_score(y_test, y_pred))

	def save_model(self, file_path):
		joblib.dump(self.linear_model, file_path)