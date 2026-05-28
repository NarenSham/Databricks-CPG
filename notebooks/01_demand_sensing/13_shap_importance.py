# 13_shap_importance.py
# Computes SHAP values for the champion model
# Stores feature importance as MLflow artifact
# Feeds the explain_prediction() agent tool

# Install required package (only works when run in notebook context)
# If running as a standalone .py file, install shap via pip in your environment first

try:
    import subprocess
    import sys
    import shap
 
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# Install xgboost if not available
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import Row, SparkSession


notebooks_dir = f'/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/Databricks-CPG/notebooks'
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Databricks-CPG/experiments/demand_sensing")

# Load champion model
model = mlflow.xgboost.load_model("models:/cpg_planning.ml.demand_model@champion")

# Load gold feature table
df = spark.table("cpg_planning.gold.demand_feature_table").toPandas()
df["ref_date"] = pd.to_datetime(df["ref_date"])

FEATURES = [
    "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
    "rolling_3m_avg", "rolling_6m_avg",
    "month", "naics_encoded", "geo_encoded"
]

df_clean = df.dropna(subset=FEATURES)
X = df_clean[FEATURES].astype("float64")

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global feature importance — mean absolute SHAP value per feature
importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("Global Feature Importance:")
print(importance)

# ── Block 2: Per-category SHAP, MLflow artifact, Gold table ──────────────────

# Attach category labels to the cleaned feature rows
# df_clean was created in block 1 — same row order as shap_values
category_labels = df_clean[["naics_code", "naics_description"]].reset_index(drop=True)

# Build a full SHAP dataframe — one row per prediction, one col per feature
shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df["naics_code"]        = category_labels["naics_code"].values
shap_df["naics_description"] = category_labels["naics_description"].values

# ── Per-category importance ───────────────────────────────────────────────────
records = []

for (code, desc), group in shap_df.groupby(["naics_code", "naics_description"]):
    for feature in FEATURES:
        records.append({
            "naics_code":        code,
            "naics_description": desc,
            "feature":           feature,
            "mean_abs_shap":     group[feature].abs().mean()
        })

category_importance = (
    pd.DataFrame(records)
    .sort_values(["naics_code", "mean_abs_shap"], ascending=[True, False])
)

print("Per-Category Feature Importance:")
print(category_importance.to_string(index=False))

# ── MLflow: log both importance tables as artifacts ───────────────────────────
with mlflow.start_run(run_name="shap_importance_v1") as run:

    # Save CSVs temporarily then log
    global_path   = "/tmp/shap_global_importance.csv"
    category_path = "/tmp/shap_category_importance.csv"

    importance.to_csv(global_path,   index=False)
    category_importance.to_csv(category_path, index=False)

    mlflow.log_artifact(global_path,   artifact_path="shap")
    mlflow.log_artifact(category_path, artifact_path="shap")

    # Log summary metrics — one metric per feature for quick scanning
    for _, row in importance.iterrows():
        mlflow.log_metric(f"shap_{row['feature']}", round(row['mean_abs_shap'], 2))

    mlflow.set_tag("model_version", "champion")
    mlflow.set_tag("artifact_type", "shap_importance")

    run_id = run.info.run_id
    print(f"\nMLflow run logged: {run_id}")

# ── Gold table: write per-category importance ─────────────────────────────────
shap_spark = spark.createDataFrame(category_importance)

(
    shap_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("cpg_planning.gold.demand_shap_importance")
)

print("Written: cpg_planning.gold.demand_shap_importance")
 
# ── Governance log ────────────────────────────────────────────────────────────
log_decision(
    agent_name       = "shap_importance",
    action      = "shap_computed",
    details     = f"TreeSHAP on champion model. Top feature: {importance.iloc[0]['feature']} "
                  f"mean_abs_shap={importance.iloc[0]['mean_abs_shap']:.0f}. "
                  f"MLflow run: {run_id}"
)

print("Governance logged.")
