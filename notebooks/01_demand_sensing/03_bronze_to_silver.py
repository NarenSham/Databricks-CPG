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