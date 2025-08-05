from src.data_loader import DatasetLoader

loader = DatasetLoader("data/housing.csv")
df = loader.load_data()
print(df.head())