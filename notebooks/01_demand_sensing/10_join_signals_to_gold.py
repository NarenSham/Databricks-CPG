
# 10_join_signals_to_gold.py
# Joins external signals to gold feature table as new columns
# Each signal becomes a column — wide format for ML

from pyspark.sql.functions import lag, col, lit
from pyspark.sql import Window


gold = spark.table("cpg_planning.gold.demand_feature_table")
signals = spark.table("cpg_planning.silver.demand_signals_monthly")

# ── PIVOT signals to wide format ──────────────────────────────
# Each signal_name becomes a column
signals_pivot = (signals
    .groupBy("ref_date", "geo", "naics_code")
    .pivot("signal_name")
    .avg("signal_value"))

# ── JOIN to gold ──────────────────────────────────────────────
# Left join — keep all gold rows, add signals where available
gold_with_signals = gold.join(
    signals_pivot,
    on=["ref_date", "geo", "naics_code"],
    how="left"
)
w = Window.partitionBy("naics_code", "geo").orderBy("ref_date")

# Lag all signal columns by 1 month to prevent leakage
signal_cols = [
    "cpi", "gas_price_cents_per_litre",
    "trends_clothing_sale_canada", "trends_costco_canada",
    "trends_fashion_canada", "trends_food_prices_canada",
    "trends_fuel_prices", "trends_gas_prices_canada",
    "trends_grocery_delivery", "trends_pharmacy_canada",
    "trends_shoppers_drug_mart", "trends_walmart_canada"
]

for signal in signal_cols:
    gold_with_signals = gold_with_signals.withColumn(
        f"{signal}_lag1",
        lag(signal, 1).over(w)
    ).drop(signal)

# Write to gold
(gold_with_signals.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("cpg_planning.gold.demand_feature_table"))

count = spark.table("cpg_planning.gold.demand_feature_table").count()
print(f"Gold feature table rows: {count}")
print(f"Total columns: {len(spark.table('cpg_planning.gold.demand_feature_table').columns)}")