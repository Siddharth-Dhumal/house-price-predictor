from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
	

def build_model_pipeline(preprocessor: ColumnTransformer, model=None) -> Pipeline:

	if model is None:
		model = LinearRegression()

	pipe = Pipeline(
		steps=[
		("pre", preprocessor),
		("model", model)
		]
	)

	return pipe 