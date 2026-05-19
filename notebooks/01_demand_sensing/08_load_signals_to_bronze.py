# 08_load_signals_to_bronze.py
# Loads all external signals into bronze.demand_signals
# One table for all signals — simpler to manage and extend

from pyspark.sql.functions import col, to_date, lit
from pyspark.sql.types import DoubleType
import pandas as pd

# Truncate and reload to avoid duplicates on re-runs
spark.sql("TRUNCATE TABLE cpg_planning.bronze.demand_signals")

TARGET_TABLE = "cpg_planning.bronze.demand_signals"



# ── CPI ──────────────────────────────────────────────────────
cpi = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/cpg_planning/bronze/landing/cpi.csv"))

cpi = (cpi
    .withColumn("ref_date", to_date(col("ref_date")))
    .withColumn("signal_value", col("cpi_value").cast(DoubleType()))
    .withColumn("naics_code", col("naics_code").cast("string"))  # ← add this
    .withColumn("signal_name", lit("cpi"))
    .withColumn("source", lit("statcan_18100004"))
    .withColumn("pulled_at", col("pulled_at").cast("timestamp"))
    .select("ref_date", "geo", "naics_code",
            "signal_name", "signal_value", "source", "pulled_at"))

# ── GAS PRICES ───────────────────────────────────────────────
gas = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/cpg_planning/bronze/landing/gas_prices.csv"))

gas = (gas
    .withColumn("ref_date", to_date(col("ref_date")))
    .withColumn("signal_value", col("signal_value").cast(DoubleType()))
    .withColumn("naics_code", col("naics_code").cast("string"))
    .withColumn("pulled_at", col("pulled_at").cast("timestamp"))
    .select("ref_date", "geo", "naics_code",
            "signal_name", "signal_value", "source", "pulled_at"))

# ── GOOGLE TRENDS ─────────────────────────────────────────────
trends = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/cpg_planning/bronze/landing/google_trends.csv"))

trends = (trends
    .withColumn("ref_date", to_date(col("ref_date")))
    .withColumn("signal_value", col("signal_value").cast(DoubleType()))
    .withColumn("naics_code", col("naics_code").cast("string"))
    .withColumn("pulled_at", col("pulled_at").cast("timestamp"))
    .select("ref_date", "geo", "naics_code",
            "signal_name", "signal_value", "source", "pulled_at"))


# Write — append so future signals accumulate
(cpi.write
    .format("delta")
    .mode("append")
    .saveAsTable(TARGET_TABLE))

print("CPI table loaded")


(gas.write
    .format("delta")
    .mode("append")
    .saveAsTable(TARGET_TABLE))

print("Gas prices loaded")

(trends.write
    .format("delta")
    .mode("append")
    .saveAsTable(TARGET_TABLE))

print("Google Trends loaded")




count = spark.table(TARGET_TABLE).count()
print(f"bronze.demand_signals rows: {count}")
display(spark.table(TARGET_TABLE)
    .groupBy("signal_name", "naics_code")
    .count()
    .orderBy("signal_name", "naics_code"))