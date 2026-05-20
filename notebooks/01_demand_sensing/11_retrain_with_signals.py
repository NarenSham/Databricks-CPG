# 11_retrain_with_signals.py
# Retrains XGBoost with external signals added
# Compares to baseline model in MLflow

import subprocess
import sys

# Install xgboost if not available
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Databricks-CPG/experiments/demand_sensing")

df = spark.table("cpg_planning.gold.demand_feature_table").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
df = df.sort_values("ref_date").reset_index(drop=True)

# All features including signals
FEATURES = [
    "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
    "rolling_3m_avg", "rolling_6m_avg",
    "month", "naics_encoded", "geo_encoded"
]


TARGET = "value"

df_clean = df.dropna(subset=FEATURES)


# # Drop rows where any signal is null
# core_signals = ["lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
#                 "rolling_3m_avg", "rolling_6m_avg",
#                 "month", "naics_encoded", "geo_encoded",
#                 "cpi_lag1", "gas_price_cents_per_litre_lag1"]

# df_clean = df.dropna(subset=core_signals)

# Fill remaining nulls with 0 — absent trend signal = no signal
# trend_cols = [c for c in df_clean.columns if c.startswith("trends_")]
# df_clean[trend_cols] = df_clean[trend_cols].fillna(0)
# print(f"Rows after dropping nulls: {len(df_clean)}")

# Time based split
cutoff = pd.to_datetime(df_clean["ref_date"].max()) - pd.DateOffset(months=12)
train = df_clean[df_clean["ref_date"] <= cutoff]
test  = df_clean[df_clean["ref_date"] > cutoff]

print(f"Train: {len(train)} | Test: {len(test)}")

X_train = train[FEATURES].astype("float64")
y_train = train[TARGET]
X_test  = test[FEATURES].astype("float64")
y_test  = test[TARGET]

with mlflow.start_run(run_name="xgboost_with_signals_v1"):

    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mape = np.mean(np.abs((y_test.values - preds) / y_test.values))

    test_copy = test.copy()
    test_copy["predicted"] = preds
    test_copy["abs_pct_error"] = np.abs(
        (test_copy["value"] - test_copy["predicted"]) / test_copy["value"]
    )
    per_category = test_copy.groupby("naics_code")["abs_pct_error"].mean()

    mlflow.log_params(params)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("train_rows", len(train))
    mlflow.log_metric("feature_count", len(FEATURES))
    for naics, cat_mape in per_category.items():
        mlflow.log_metric(f"{naics}_mape", cat_mape)

    mlflow.xgboost.log_model(
        model,
        name="model",
        input_example=X_train.iloc[:5]
    )

    print(f"Overall MAPE: {mape:.4f}")
    print(f"Previous MAPE: 0.0468")
    print(f"Improvement: {(0.0468 - mape) / 0.0468:.1%}")
    print("\nPer category MAPE:")
    print(per_category)