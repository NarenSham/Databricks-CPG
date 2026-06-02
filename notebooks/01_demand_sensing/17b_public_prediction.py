# ── First Public Prediction — July 2026 ──────────────────────────────────────
# Using the latest available feature row (Feb 2026) to predict March 2026
# This is the first timestamped public commitment from this system

import json
import sys
from datetime import datetime
import pandas as pd
import mlflow.xgboost
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])


# Load accuracy table for confidence ranges
accuracy = spark.table("cpg_planning.gold.model_accuracy").toPandas()
mape_dict = dict(zip(accuracy["naics_code"], accuracy["mape"]))

# Get latest feature row per category + province
latest = (
    spark.table("cpg_planning.gold.demand_feature_table")
    .toPandas()
)
latest["ref_date"] = pd.to_datetime(latest["ref_date"])
latest = (
    latest.sort_values("ref_date")
    .groupby(["naics_code", "geo"])
    .last()
    .reset_index()
)

FEATURES = [
    "lag_1m", "lag_2m", "lag_3m", "lag_6m", "lag_12m",
    "rolling_3m_avg", "rolling_6m_avg",
    "month", "naics_encoded", "geo_encoded"
]

CATEGORY_NAMES = {
    "445": "Food and Beverage",
    "455": "General Merchandise",
    "456": "Health and Personal Care",
    "457": "Gasoline Stations",
    "458": "Clothing and Accessories",
}

CONFIDENCE_LEVEL = {
    "445": "high",
    "455": "high",
    "456": "high",
    "457": "moderate",
    "458": "low",
}

# Load champion model
model = mlflow.xgboost.load_model(
    "models:/cpg_planning.ml.demand_model@champion"
)
print("Champion model loaded.")

# Generate predictions
predictions = []
for _, row in latest.iterrows():
    X = pd.DataFrame([row[FEATURES]]).astype("float64")
    pred = float(model.predict(X)[0])
    
    naics   = str(row["naics_code"])
    mape    = mape_dict.get(naics, 0.0468)
    lower   = pred * (1 - mape)
    upper   = pred * (1 + mape)

    # Convert from thousands to billions
    def to_billions(v): return round(v * 1000 / 1e9, 3)

    predictions.append({
        "category":          CATEGORY_NAMES.get(naics, naics),
        "naics_code":        naics,
        "province":          row["geo"],
        "forecast_billions": to_billions(pred),
        "low_billions":      to_billions(lower),
        "high_billions":     to_billions(upper),
        "confidence":        CONFIDENCE_LEVEL.get(naics, "moderate"),
        "forecast_error_pct": f"±{round(mape * 100, 1)}%",
        "primary_driver":    "year-over-year seasonality"
    })

# Build the full prediction document
output = {
    "meta": {
        "prediction_date":     datetime.now().strftime("%Y-%m-%d"),
        "forecast_month":      "2026-07-01",
        "data_through":        "2026-02-01",
        "model":               "cpg_planning.ml.demand_model@champion",
        "backtest_mape":       "4.85%",
        "note": (
            "First public prediction from CPG Compound AI Platform. "
            "Actuals will be published by Statistics Canada ~August 2026. "
            "Track record: gold.demand_predictions (Jan 2025 - Feb 2026)"
        )
    },
    "predictions": predictions
}

# Print summary
print(f"Predictions: {len(predictions)}")
print(f"Forecast month: {output['meta']['forecast_month']}")
print(f"Generated: {output['meta']['prediction_date']}\n")

# Show Ontario Food as sanity check
ontario_food = [
    p for p in predictions
    if p["province"] == "Ontario" and p["naics_code"] == "445"
][0]
print(f"Ontario Food: ${ontario_food['forecast_billions']}B "
      f"({ontario_food['low_billions']}–{ontario_food['high_billions']}B) "
      f"{ontario_food['confidence']} confidence")

# Save to file
username = spark.sql("SELECT current_user()").collect()[0][0]
output_path = f"/Workspace/Users/{username}/Databricks-CPG/predictions/prediction_2026_07.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {output_path}")