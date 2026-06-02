

# 18_lakeflow_job.py
# Orchestrates the full monthly demand sensing pipeline
# North star: feeds the demand planning app with fresh forecasts
# and override accuracy scores every month
#
# Schedule: 1st of month, 6am Toronto time
# Trigger: manual or scheduled after StatCan CSV is pushed to Volume

import sys
import runpy
from datetime import datetime
import mlflow

notebooks_dir = (
    f'/Workspace/Users/'
    f'{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}'
    f'/Databricks-CPG/notebooks'
)
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision



username    = spark.sql("SELECT current_user()").collect()[0][0]
BASE_PATH   = f"/Workspace/Users/{username}/Databricks-CPG/notebooks/01_demand_sensing"
RUN_DATE    = datetime.now().strftime("%Y-%m-%d")

mlflow.set_experiment(
    f"/Users/{username}/Databricks-CPG/experiments/demand_sensing"
)

print(f"Pipeline start: {RUN_DATE}")
print(f"{'─'*60}")

# ── Helper ────────────────────────────────────────────────────────────────────
def run_python_file(name: str):
    """
    Executes a Python file and logs success or failure.
    Raises on failure so the DAG stops cleanly.
    """
    path = f"{BASE_PATH}/{name}"
    print(f"\n▶ Running: {name}")
    try:
        # Execute the Python file with access to current globals (spark, dbutils, etc.)
        runpy.run_path(path, init_globals=globals(), run_name="__main__")
        print(f"✅ {name} complete")
        log_decision(
            agent_name = "18_lakeflow_job",
            action     = f"script_success:{name}",
            details    = f"run_date={RUN_DATE}"
        )
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        log_decision(
            agent_name = "18_lakeflow_job",
            action     = f"script_failed:{name}",
            details    = f"run_date={RUN_DATE} error={str(e)[:200]}"
        )
        raise  # stops the pipeline


# ── Check if retrain is needed ────────────────────────────────────────────────
def should_retrain() -> bool:
    """
    Checks recent backtest MAPE against training MAPE.
    Returns True if recent performance has degraded by more than 20%.
    """
    try:
        from pyspark.sql.functions import avg
        recent = (
            spark.table("cpg_planning.gold.demand_predictions")
            .filter("ref_date >= add_months(current_date(), -3)")
            .agg(avg("pct_error").alias("recent_mape"))
            .collect()[0]["recent_mape"]
        )
        training_mape = 4.68
        degraded      = recent > training_mape * 1.20

        print(f"Recent MAPE (3mo): {recent:.2f}% | Training MAPE: {training_mape}%")
        print(f"Retrain needed: {degraded}")
        return degraded

    except Exception as e:
        print(f"Could not assess retrain need: {e}. Skipping retrain.")
        return False


# ── Pipeline ──────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name=f"monthly_pipeline_{RUN_DATE}"):

    # Stage 1 — Data ingestion
    print("\n── STAGE 1: Data Ingestion ───────────────────────────────")
    run_python_file("01_load_statcan_to_bronze.py")
    run_python_file("08_load_signals_to_bronze.py")

    # Stage 2 — Transformation
    print("\n── STAGE 2: Transformation ───────────────────────────────")
    run_python_file("03_bronze_to_silver.py")
    run_python_file("09_signals_to_silver.py")
    run_python_file("04_silver_to_gold.py")

    # Stage 3 — Model (conditional retrain)
    print("\n── STAGE 3: Model ────────────────────────────────────────")
    if should_retrain():
        print("Retraining triggered by MAPE degradation.")
        run_python_file("06_train_model.py")
        run_python_file("07_register_model.py")
        run_python_file("13_shap_importance.py")
    else:
        print("Model healthy — skipping retrain.")

    # Stage 4 — Backtest and predictions
    print("\n── STAGE 4: Backtest + Predictions ──────────────────────")
    run_python_file("17_backtest.py")

    # Stage 5 — Evaluation
    print("\n── STAGE 5: Agent Evaluation ────────────────────────────")
    run_python_file("16_evaluate_agent.py")

    # Stage 6 — Override scoring (feeds the app)
    # Notebook 18b scores last month's human overrides against actuals
    # Skip gracefully if no overrides exist yet
    print("\n── STAGE 6: Override Scoring ────────────────────────────")
    try:
        override_count = (
            spark.table("cpg_planning.gold.demand_overrides")
            .filter("actual_value IS NULL")
            .count()
        )
        if override_count > 0:
            run_python_file("18b_score_overrides.py")
        else:
            print("No pending overrides to score — skipping.")
    except Exception:
        print("Override table not yet created — skipping. "
              "Will activate after app is built.")

    # ── Log pipeline summary ──────────────────────────────────────────────────
    mlflow.set_tag("pipeline_run_date", RUN_DATE)
    mlflow.set_tag("pipeline_type",     "monthly_demand_sensing")

    print(f"\n{'═'*60}")
    print(f"Pipeline complete: {RUN_DATE}")
    print(f"{'═'*60}")

    log_decision(
        agent_name = "18_lakeflow_job",
        action     = "pipeline_complete",
        details    = f"run_date={RUN_DATE}"
    )

print("\n18_ complete. App has fresh forecasts.")
