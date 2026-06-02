# Databricks notebook source

# COMMAND ----------
# notebooks/01_demand_sensing/06_train_model.py
# Trains a pooled XGBoost model across all categories and provinces.
# Uses time-series cross validation — never random split.

import subprocess
import sys
# Add the workspace notebooks directory to the Python path
notebooks_dir = f'/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/Databricks-CPG/notebooks'
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision
# Install xgboost if not available
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Databricks-CPG/experiments/demand_sensing")

# Load gold table
df = spark.table("cpg_planning.gold.demand_feature_table").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
df = df.sort_values("ref_date").reset_index(drop=True)

# Define features and target
FEATURES = [
    "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
    "rolling_3m_avg", "rolling_6m_avg",
    "month", "naics_encoded", "geo_encoded"
]
TARGET = "value"

# Time-based split — last 12 months as holdout
# Never random split on time series data
cutoff = pd.to_datetime(df["ref_date"].max()) - pd.DateOffset(months=12)
train = df[df["ref_date"] <= cutoff]
test  = df[df["ref_date"] > cutoff]

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Train period: {train['ref_date'].min()} to {train['ref_date'].max()}")
print(f"Test period:  {test['ref_date'].min()} to {test['ref_date'].max()}")

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# Cast features to float64 to handle missing values cleanly
X_train = X_train.astype("float64")
X_test = X_test.astype("float64")



with mlflow.start_run(run_name="xgboost_pooled_v1", nested=True):
    
    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predict on test set
    preds = model.predict(X_test)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test.values - preds) / y_test.values))
    
    # Per category MAPE
    test_copy = test.copy()
    test_copy["predicted"] = preds
    test_copy["abs_pct_error"] = np.abs(
        (test_copy["value"] - test_copy["predicted"]) / test_copy["value"]
    )
    per_category = test_copy.groupby("naics_code")["abs_pct_error"].mean()
    
    # Log to MLflow
    mlflow.xgboost.log_model(
    model, 
    name = "model",
    input_example=X_train.iloc[:5]
    )
    mlflow.log_params(params)
    mlflow.log_metric("mape", mape)
    for naics, cat_mape in per_category.items():
        mlflow.log_metric(f"{naics}_mape", cat_mape)
    
    print(f"Overall MAPE: {mape:.4f}")
    print("\nPer category MAPE:")
    print(per_category)

# ── Write model accuracy to gold table ───────────────────────────────────
    # Single source of truth for agent confidence calibration
    # Replaces hardcoded MAPE dicts in downstream notebooks
    
    CATEGORY_NAMES_MAP = {
        "445": "Food and Beverage",
        "455": "General Merchandise",
        "456": "Health and Personal Care",
        "457": "Gasoline Stations",
        "458": "Clothing and Accessories",
    }
    
    accuracy_records = [
        {
            "naics_code":    str(naics),
            "category_name": CATEGORY_NAMES_MAP.get(str(naics), str(naics)),
            "mape":          float(cat_mape),
            "model_version": "champion",
            "run_id":        mlflow.active_run().info.run_id,
        }
        for naics, cat_mape in per_category.items()
    ]
    
    (spark.createDataFrame(accuracy_records)
     .write.format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable("cpg_planning.gold.model_accuracy"))
    
    print("\nModel accuracy table written: cpg_planning.gold.model_accuracy")


log_decision(
    agent_name="demand_agent",
    action="model_trained",
    details=f"XGBoost pooled model. Overall MAPE: {mape:.4f}. Features: {FEATURES}. Train rows: {len(train)}. Test rows: {len(test)}."
)

