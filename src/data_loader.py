import pandas as pd

class DatasetLoader:
	def __init__(self, file_path):
		self.file_path = file_path

	def load_data(self):
		try:
			data = pd.read_csv(self.file_path)
			return data

		except FileNotFoundError:
			print(f"[ERROR] File not found at: {self.file_path}")

			return None

		except Exception as e:
			print(f"[ERROR] An error has occurred: {e}")

			return None