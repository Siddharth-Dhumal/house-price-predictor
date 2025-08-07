from src.data_loader import DatasetLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

loader = DatasetLoader("data/housing.csv")
print("Loading data")
df = loader.load_data()
print("Data loaded successfully")
print(df.head())

feature_eng = FeatureEngineer()
X_train, X_test, y_train, y_test = feature_eng.transform(df)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

trainer = ModelTrainer()
print("Training model")
trainer.train(X_train, y_train)
print("Model training complete")
print("Evaluating model")
print(trainer.evaluate(X_test, y_test))
print("Model evaluation complete")
trainer.save_model("models/housing_model.pkl")