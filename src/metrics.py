from typing import Dict
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate

def evaluate_holdout(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
	y_pred = pipe.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	epsilon = 1e-8
	y_true_arr = np.array(y_test)
	y_pred_arr = np.array(y_pred)
	denominator = np.where(y_true_arr == 0, epsilon, y_true_arr)
	mape = np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100

	return {
		"test_rmse": float(rmse),
		"test_mae": float(mae),
		"test_r2": float(r2),
		"test_mape": float(mape),
	}

def baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
	mean_train = float(np.mean(y_train))
	y_hat = np.full_like(y_test, mean_train, dtype=float)

	rmse = np.sqrt(mean_squared_error(y_test, y_hat))
	mae = mean_absolute_error(y_test, y_hat)
	r2 = r2_score(y_test, y_hat)

	epsilon = 1e-8
	y_true_arr = np.array(y_test)
	y_pred_arr = np.array(y_hat)
	denominator = np.where(y_true_arr == 0, epsilon, y_true_arr)
	mape = np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100

	return {
		"baseline_rmse": float(rmse),
		"baseline_mae": float(mae),
		"baseline_r2": float(r2),
		"baseline_mape": float(mape),
	}

def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
	scoring = {
		"rmse": "neg_root_mean_squared_error",
		"mae":  "neg_mean_absolute_error",
		"r2": "r2",
		"mape": "neg_mean_absolute_percentage_error",
	}

	cv_res = cross_validate(
		estimator=pipe,
		X=X,
		y=y,
		cv=cv,
		scoring=scoring,
		return_train_score=False,
		n_jobs=None,
	)

	rmse = -cv_res["test_rmse"]
	mae = -cv_res["test_mae"]
	r2 = cv_res["test_r2"]
	mape = -cv_res["test_mape"] * 100

	return {
		"cv_rmse_mean": float(np.mean(rmse)),
        "cv_rmse_std":  float(np.std(rmse)),
        "cv_mae_mean":  float(np.mean(mae)),
        "cv_mae_std":   float(np.std(mae)),
        "cv_r2_mean":   float(np.mean(r2)),
        "cv_r2_std":    float(np.std(r2)),
        "cv_mape_mean": float(np.mean(mape)),
        "cv_mape_std":  float(np.std(mape)),
    }