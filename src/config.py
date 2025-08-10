from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class Config:
	data_path: str
	model_path: str
	reports_dir: str
	test_size: float
	random_state: int
	target: str
	numerical: List[str]
	categorical: List[str]

	def load_config(path: str) -> Config:
		with open(path, "r") as f:
			cfg = yaml.safe_load(f)

			return Config(
			    data_path=cfg["paths"]["data"],
			    model_path=cfg["paths"]["model"],
			    reports_dir=cfg["paths"]["reports"],
			    test_size=float(cfg["split"]["test_size"]),
			    random_state=int(cfg["split"]["random_state"]),
			    target=cfg["target"],
			    numerical=list(cfg["numerical"]),
			    categorical=list(cfg["categorical"]),
			)