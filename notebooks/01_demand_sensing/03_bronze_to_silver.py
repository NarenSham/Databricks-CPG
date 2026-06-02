
# Databricks notebook source

# COMMAND ----------

import sys

# Add the workspace notebooks directory to the Python path
notebooks_dir = f'/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/Databricks-CPG/notebooks'
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision

from pyspark.sql.functions import col, sum as spark_sum

TARGET_NAICS = ["445", "456", "457", "458", "455"]

MAJOR_PROVINCES = [
    "Ontario", "Quebec", "British Columbia",
    "Alberta", "Manitoba", "Saskatchewan"
]

CLEAN_STATUSES = ["A", "B", "C", "D", "E", None]

df = spark.table("cpg_planning.bronze.demand_statcan_retail")

silver = (df
    .filter(col("naics_code").isin(TARGET_NAICS))
    .filter(col("geo").isin(MAJOR_PROVINCES))
    .filter(col("status").isin(CLEAN_STATUSES) | col("status").isNull())
    .select("ref_date", "geo", "naics_code", "naics_description", "value", "status"))

(silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("cpg_planning.silver.demand_retail_monthly"))

# Validate
count = spark.table("cpg_planning.silver.demand_retail_monthly").count()
print(f"Silver table rows: {count}")

log_decision(
    agent_name="system",
    action="silver_table_refreshed",
    details=f"bronze → silver complete. Rows: {count}. Categories: 445,455,456,457,458. Provinces: 6 major."
)
