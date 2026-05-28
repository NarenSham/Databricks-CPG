# 14_create_agent_tools.py
# Registers UC SQL functions as agent tools in cpg_planning.ml
# Only tools that are pure SELECT against Delta tables belong here
# Compute tools (predict, explain via model) live in 15_build_demand_agent.py

import sys
notebooks_dir = f'/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/Databricks-CPG/notebooks'
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision

# ── TOOL 1: get_data_freshness ────────────────────────────────────────────────
# Returns the latest available data date across silver and gold
# Agent uses this to answer: "how current is your data?"

spark.sql("""
CREATE OR REPLACE FUNCTION cpg_planning.ml.get_data_freshness()
RETURNS TABLE(layer STRING, latest_date DATE, row_count BIGINT)
COMMENT 'Returns the latest ref_date and row count for silver and gold demand tables. Use to assess data currency before making predictions.'
RETURN
  SELECT 'silver' AS layer,
         MAX(ref_date) AS latest_date,
         COUNT(*)      AS row_count
  FROM cpg_planning.silver.demand_retail_monthly

  UNION ALL

  SELECT 'gold'   AS layer,
         MAX(ref_date) AS latest_date,
         COUNT(*)      AS row_count
  FROM cpg_planning.gold.demand_feature_table
""")

print("Tool 1 registered: cpg_planning.ml.get_data_freshness")

# ── TOOL 2: explain_prediction ────────────────────────────────────────────────
# Returns top feature drivers for a given category from the SHAP importance table
# Agent uses this to answer: "why did the model predict X for Food in Ontario?"
# Note: this explains the category-level driver profile, not a single row
# Row-level SHAP would require model inference — that stays in Python tools

spark.sql("""
CREATE OR REPLACE FUNCTION cpg_planning.ml.explain_prediction(
  category STRING,
  geo      STRING
)
RETURNS TABLE(feature STRING, mean_abs_shap DOUBLE, rank BIGINT)
COMMENT 'Returns ranked feature importance for a given NAICS category from SHAP analysis of the champion model. Category-level explanation, not row-level.'
RETURN
  SELECT
    feature,
    mean_abs_shap,
    ROW_NUMBER() OVER (ORDER BY mean_abs_shap DESC) AS rank
  FROM cpg_planning.gold.demand_shap_importance
  WHERE naics_code = category
  ORDER BY mean_abs_shap DESC
""")

print("Tool 2 registered: cpg_planning.ml.explain_prediction")

# ── Smoke test both tools ─────────────────────────────────────────────────────
print("\n── Smoke Test: get_data_freshness ──")
spark.sql("SELECT * FROM cpg_planning.ml.get_data_freshness()").show()

print("── Smoke Test: explain_prediction (Food / Ontario) ──")
spark.sql("SELECT * FROM cpg_planning.ml.explain_prediction('445', 'Ontario')").show()

# ── Governance log ────────────────────────────────────────────────────────────
log_decision(
    agent_name = "14_create_agent_tools",
    action     = "uc_sql_tools_registered",
    details    = "Registered 2 UC SQL functions: get_data_freshness, explain_prediction. "
                 "Compute tools (predict_next_month, compare_to_baseline, get_accuracy_history) "
                 "deferred to Python tools in 15_build_demand_agent.py — require MLflow/Spark access "
                 "unavailable in UC function sandbox."
)

print("\nGovernance logged.")
