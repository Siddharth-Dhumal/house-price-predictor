from src.data_loader import DatasetLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

loader = DatasetLoader("data/housing.csv")
print("Loading data")
df = loader.load_data()
print("Data loaded successfully")
print(df.head())

feature_eng = FeatureEngineer()
X_train, X_test, y_train, y_test, feature_names = feature_eng.transform(df)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

trainer = ModelTrainer()
print("Training model")
trainer.train(X_train, y_train)
print("Model training complete")
print("Evaluating model")
trainer.evaluate(X_test, y_test)
print("Model evaluation complete")
trainer.save_model("models/housing_model.pkl")

new_house = {
    "longitude": -122.25,
    "latitude": 37.85,
    "housing_median_age": 42.0,
    "total_rooms": 1467.0,
    "total_bedrooms": 190.0,
    "population": 496.0,
    "households": 177.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}

predictor = Predictor("models/housing_model.pkl", feature_names)
predicted_price = predictor.predict(new_house)
print("The predicted price of the house is", predicted_price)