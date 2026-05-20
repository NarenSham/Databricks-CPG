# 09_signals_to_silver.py
# Cleans and aligns external signals to monthly grain
# Handles province broadcast for Canada-level signals

from pyspark.sql.functions import col, avg, lit
from pyspark.sql import Window

MAJOR_PROVINCES = [
    "Ontario", "Quebec", "British Columbia",
    "Alberta", "Manitoba", "Saskatchewan"
]

spark.sql("TRUNCATE TABLE cpg_planning.silver.demand_signals_monthly")


signals = spark.table("cpg_planning.bronze.demand_signals")

# ── CPI + GAS: already provincial, just clean ─────────────────
provincial_signals = signals.filter(col("geo") != "Canada")

# ── GOOGLE TRENDS: Canada-level → broadcast to all provinces ──
trends = signals.filter(col("geo") == "Canada")

# Cross join with provinces to replicate national signal
provinces_df = spark.createDataFrame(
    [(p,) for p in MAJOR_PROVINCES], ["geo"]
)

trends_broadcast = (trends
    .drop("geo")
    .crossJoin(provinces_df))

# ── COMBINE ───────────────────────────────────────────────────
silver_signals = provincial_signals.unionByName(trends_broadcast)

(silver_signals.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("cpg_planning.silver.demand_signals_monthly"))

count = spark.table("cpg_planning.silver.demand_signals_monthly").count()
print(f"Silver signals rows: {count}")