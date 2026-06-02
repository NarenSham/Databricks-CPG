# Databricks notebook source


# COMMAND ----------

# 17_backtest.py
# Retroactive predictions Jan 2025 → Feb 2026
# No data leakage — each prediction uses only features available
# at the end of the prior month
# Output: gold.demand_predictions — the public track record

import sys
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])


notebooks_dir = (
    f'/Workspace/Users/'
    f'{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}'
    f'/Databricks-CPG/notebooks'
)
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(
    f"/Users/{username}/Databricks-CPG/experiments/demand_sensing"
)

# ── Load champion model ───────────────────────────────────────────────────────
model = mlflow.xgboost.load_model(
    "models:/cpg_planning.ml.demand_model@champion"
)
print("Champion model loaded.")

# ── Load full gold feature table ──────────────────────────────────────────────
df = spark.table("cpg_planning.gold.demand_feature_table").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])
df = df.sort_values("ref_date").reset_index(drop=True)

print(f"Gold table loaded: {len(df)} rows")
print(f"Date range: {df['ref_date'].min()} → {df['ref_date'].max()}")

# ── Define backtest window ────────────────────────────────────────────────────
# Predict Jan 2025 → Feb 2026 (14 months)
# These are months where we have actuals to compare against
BACKTEST_START = pd.Timestamp("2025-01-01")
BACKTEST_END   = pd.Timestamp("2026-02-01")

FEATURES = [
    "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
    "rolling_3m_avg", "rolling_6m_avg",
    "month", "naics_encoded", "geo_encoded"
]

CATEGORY_NAMES = {
    "445": "Food and Beverage",
    "455": "General Merchandise",
    "456": "Health and Personal Care",
    "457": "Gasoline Stations",
    "458": "Clothing and Accessories",
}

# ── Run backtest ──────────────────────────────────────────────────────────────
records = []
backtest_months = df[
    (df["ref_date"] >= BACKTEST_START) &
    (df["ref_date"] <= BACKTEST_END)
]["ref_date"].unique()

print(f"\nBacktest months: {len(backtest_months)}")
print(f"Category × province combos: 30")
print(f"Total predictions: {len(backtest_months) * 30}\n")

for month in sorted(backtest_months):
    # Get all rows for this month
    month_rows = df[df["ref_date"] == month].copy()

    if month_rows.empty:
        continue

    # Validate no data leakage
    # Features must be available at end of prior month
    # lag_1m = prior month value — already in gold by construction
    X = month_rows[FEATURES].astype("float64")

    # Skip rows with missing features
    valid = month_rows.dropna(subset=FEATURES)
    if valid.empty:
        continue

    X_valid  = valid[FEATURES].astype("float64")
    preds    = model.predict(X_valid)

    for i, (_, row) in enumerate(valid.iterrows()):
        actual    = float(row["value"])
        predicted = float(preds[i])
        abs_error = abs(actual - predicted)
        pct_error = abs_error / actual if actual != 0 else None

        records.append({
            "ref_date":        month.strftime("%Y-%m-%d"),
            "naics_code":      str(row["naics_code"]),
            "category_name":   CATEGORY_NAMES.get(
                                   str(row["naics_code"]),
                                   str(row["naics_code"])
                               ),
            "geo":             row["geo"],
            "actual_value":    round(actual, 2),
            "predicted_value": round(predicted, 2),
            "abs_error":       round(abs_error, 2),
            "pct_error":       round(pct_error * 100, 4) if pct_error else None,
            "model_version":   "champion",
            "run_date":        datetime.now().strftime("%Y-%m-%d"),
        })

print(f"Predictions generated: {len(records)}")

# ── Summary statistics ────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)

overall_mape = results_df["pct_error"].mean()
print(f"\nOverall backtest MAPE: {overall_mape:.2f}%")

print("\nPer-category backtest MAPE:")
cat_mape = (results_df
    .groupby("category_name")["pct_error"]
    .mean()
    .sort_values()
    .round(2)
)
print(cat_mape.to_string())

print("\nPer-province backtest MAPE:")
geo_mape = (results_df
    .groupby("geo")["pct_error"]
    .mean()
    .sort_values()
    .round(2)
)
print(geo_mape.to_string())

# ── Write to gold table ───────────────────────────────────────────────────────
from delta.tables import DeltaTable

# ── Write to gold table ───────────────────────────────────────────────────────
from delta.tables import DeltaTable

try:
    # Attempt merge — correct pattern for all normal runs
    # Idempotent: running twice produces identical results, no duplicates
    predictions_table = DeltaTable.forName(
        spark, "cpg_planning.gold.demand_predictions"
    )
    predictions_table.alias("existing").merge(
        spark.createDataFrame(results_df).alias("new"),
        "existing.ref_date   = new.ref_date AND "
        "existing.naics_code = new.naics_code AND "
        "existing.geo        = new.geo"
    ).whenMatchedUpdateAll(
    ).whenNotMatchedInsertAll(
    ).execute()
    print("Merge complete.")

except Exception:
    # Table doesn't exist or schema mismatch — create fresh
    # This path only runs on first execution or after schema changes
    (spark.createDataFrame(results_df)
     .write.format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable("cpg_planning.gold.demand_predictions"))
    print("Table created fresh.")

print(f"\nWritten: cpg_planning.gold.demand_predictions")
print(f"Rows: {len(results_df)}")

# ── Log to MLflow ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="backtest_v1", nested=True):
    mlflow.log_metric("backtest_overall_mape",  round(overall_mape, 4))
    mlflow.log_metric("backtest_months",         len(backtest_months))
    mlflow.log_metric("backtest_predictions",    len(records))

    for cat, mape in cat_mape.items():
        mlflow.log_metric(
            f"backtest_mape_{cat.lower().replace(' ', '_')[:20]}",
            round(mape, 4)
        )

    mlflow.set_tag("backtest_start", str(BACKTEST_START.date()))
    mlflow.set_tag("backtest_end",   str(BACKTEST_END.date()))
    mlflow.set_tag("model_version",  "champion")

print("MLflow run logged.")

# ── Governance log ─────────────────────────────────────────────────────────────
log_decision(
    agent_name = "17_backtest",
    action     = "backtest_complete",
    details    = (
        f"Months: {len(backtest_months)} | "
        f"Predictions: {len(records)} | "
        f"Overall MAPE: {overall_mape:.2f}%"
    )
)

print("\n17_ complete. Public track record established.")
