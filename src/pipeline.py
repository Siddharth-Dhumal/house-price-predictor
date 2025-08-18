from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def build_preprocessor(numerical: List[str], categorical: List[str]) -> ColumnTransformer:

	num_branch = Pipeline(
		steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])

	cat_branch = Pipeline(
		steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
		("num",num_branch,numerical),
		("cat",cat_branch,categorical),
		],
		remainder="drop",
		verbose_feature_names_out=False,
	)

	return preprocessor
	
def make_model(name: str, params: dict):
	name = (name or "LinearRegression").lower()
	if name == "linearregression":
		return LinearRegression(**params)
	if name == "randomforestregressor":
		return RandomForestRegressor(**params)
	if name == "gradientboostingregressor":
		return GradientBoostingRegressor(**params)
	raise ValueError(f"Unknown model type: {name}")

def build_model_pipeline(preprocessor: ColumnTransformer, model=None, cfg_model: dict | None = None) -> Pipeline:

	if model is None:
		if not cfg_model:
			model = LinearRegression()
		else:
			model = make_model(cfg_model.get("type"), cfg_model.get("params", {}))

	pipe = Pipeline(
		steps=[
		("pre", preprocessor),
		("model", model)
		]
	)

	return pipe 