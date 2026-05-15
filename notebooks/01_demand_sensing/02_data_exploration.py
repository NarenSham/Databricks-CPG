df = spark.table("cpg_planning.bronze.demand_statcan_retail")

# Shape of the data
print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")
print(f"Date range:")
display(df.selectExpr("min(ref_date)", "max(ref_date)"))

# How many rows per NAICS code
print("Rows per NAICS code:")
display(df.groupBy("naics_code", "naics_description")
    .count()
    .orderBy("naics_code"))

# How many suppressed values (status = 'x')
print("Suppressed values:")
display(df.groupBy("status").count())