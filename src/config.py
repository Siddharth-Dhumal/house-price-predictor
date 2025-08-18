from dataclasses import dataclass
from typing import List, Dict, Any
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
    models_default: str
    models_presets: Dict[str, Dict[str, Any]]

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", {}) or {}
    presets = models.get("presets", {}) or {}
    default_name = models.get("default") or (next(iter(presets)) if presets else "LinearRegression")

    return Config(
        data_path=cfg["paths"]["data"],
        model_path=cfg["paths"]["model"],
        reports_dir=cfg["paths"]["reports"],
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        target=cfg["target"],
        numerical=list(cfg["numerical"]),
        categorical=list(cfg["categorical"]),
        models_default=default_name,
        models_presets=presets,
        )