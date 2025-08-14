# House Price Predictor

End-to-end ML pipeline to predict California housing prices.

## 🚀 Features
- **Config-driven** setup with YAML for easy reproducibility
- **Clean modular codebase** under `src/` and `scripts/`
- **Automated preprocessing pipeline**:
  - Missing value imputation (median & most frequent)
  - Standard scaling for numeric features
  - One-hot encoding for categorical features
- **Cross-validation & holdout evaluation** with RMSE, MAE, R²
- **Baseline metrics** for comparison
- **Timestamped prediction reports** in CSV format
- **Strict input validation** for predictions