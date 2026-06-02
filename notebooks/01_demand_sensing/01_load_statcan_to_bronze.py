# Databricks notebook source

# COMMAND ----------

# notebooks/01_demand_sensing/01_load_statcan_to_bronze.py
# Reads StatCan retail CSV from landing Volume → writes to bronze Delta table.

from pyspark.sql.functions import col, to_date

VOLUME_PATH = "/Volumes/cpg_planning/bronze/landing/statcan_retail.csv"
TARGET_TABLE = "cpg_planning.bronze.demand_statcan_retail"

# Read CSV
df = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(VOLUME_PATH))

# Cast columns to correct types
df = (df
    .withColumn("ref_date", to_date(col("ref_date"), "yyyy-MM-dd"))
    .withColumn("value", col("value").cast("double"))
    .withColumn("naics_code", col("naics_code").cast("string"))
    .withColumn("pulled_at", col("pulled_at").cast("timestamp")))

# Write to bronze — append so future refreshes accumulate
(df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE))

# Validate
count = spark.table(TARGET_TABLE).count()
print(f"Loaded {count} rows into {TARGET_TABLE}")

display(spark.table(TARGET_TABLE).limit(5))