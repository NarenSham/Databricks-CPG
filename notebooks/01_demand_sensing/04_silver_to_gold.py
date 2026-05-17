# notebooks/01_demand_sensing/04_silver_to_gold.py
# Builds gold feature table from silver.
# All features lagged to prevent leakage — when predicting month T,
# only data available through month T-1 is used.

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, lag, avg, month, when, round as spark_round
)

df = spark.table("cpg_planning.silver.demand_retail_monthly")

# Window partitioned by category + province, ordered by time
w = Window.partitionBy("naics_code", "geo").orderBy("ref_date")

features = (df
    .withColumn("lag_1m",  lag("value", 1).over(w))
    .withColumn("lag_2m",  lag("value", 2).over(w))
    .withColumn("lag_3m",  lag("value", 3).over(w))
    .withColumn("lag_6m",  lag("value", 6).over(w))
    .withColumn("lag_12m", lag("value", 12).over(w))
    .withColumn("rolling_3m_avg",
        spark_round(avg("value").over(w.rowsBetween(-3, -1)), 2))
    .withColumn("rolling_6m_avg",
        spark_round(avg("value").over(w.rowsBetween(-6, -1)), 2))
)

# Drop rows where lags are null (first 12 months per partition)
features = features.dropna(subset=["lag_12m"]) 

features = (features
    .withColumn("month", month(col("ref_date")))
    .withColumn("naics_encoded", 
        when(col("naics_code") == "445", 1)
        .when(col("naics_code") == "455", 2)
        .when(col("naics_code") == "456", 3)
        .when(col("naics_code") == "457", 4)
        .when(col("naics_code") == "458", 5))
    .withColumn("geo_encoded",
        when(col("geo") == "Ontario", 1)
        .when(col("geo") == "Quebec", 2)
        .when(col("geo") == "British Columbia", 3)
        .when(col("geo") == "Alberta", 4)
        .when(col("geo") == "Manitoba", 5)
        .when(col("geo") == "Saskatchewan", 6)))

(features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("cpg_planning.gold.demand_feature_table"))

count = spark.table("cpg_planning.gold.demand_feature_table").count()
print(f"Gold feature table rows: {count}")

print(f"Feature rows: {features.count()}")
display(features.limit(5))