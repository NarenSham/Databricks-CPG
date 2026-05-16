# notebooks/01_demand_sensing/05_baselines.py
# Three naive baselines logged to MLflow.
# Every future model must beat these to be considered an improvement.

import mlflow
import mlflow.sklearn
from pyspark.sql.functions import col, abs as spark_abs, avg

# Get current username programmatically
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/Databricks-CPG/experiments/demand_sensing")




df = spark.table("cpg_planning.gold.demand_feature_table")

def compute_mape(df, prediction_col):
    return (df
        .withColumn("abs_pct_error",
            spark_abs((col("value") - col(prediction_col)) / col("value")))
        .agg(avg("abs_pct_error"))
        .first()[0])

# --- Baseline 1: Last Month ---
with mlflow.start_run(run_name="baseline_last_month"):
    mape = compute_mape(df, "lag_1m")
    mlflow.log_metric("mape", mape)
    mlflow.log_param("model_type", "naive_last_month")
    mlflow.log_param("prediction_col", "lag_1m")
    print(f"Last month baseline MAPE: {mape:.4f}")


# --- Baseline 2: Same Month Last Year ---
with mlflow.start_run(run_name="baseline_same_month_last_year"):
    mape = compute_mape(df, "lag_12m")
    mlflow.log_metric("mape", mape)
    mlflow.log_param("model_type", "naive_same_month_last_year")
    mlflow.log_param("prediction_col", "lag_12m")
    print(f"Same month last year MAPE: {mape:.4f}")

# --- Baseline 3: Rolling 3 Month Average ---
with mlflow.start_run(run_name="baseline_rolling_3m"):
    mape = compute_mape(df, "rolling_3m_avg")
    mlflow.log_metric("mape", mape)
    mlflow.log_param("model_type", "naive_rolling_3m")
    mlflow.log_param("prediction_col", "rolling_3m_avg")
    print(f"Rolling 3m average MAPE: {mape:.4f}")