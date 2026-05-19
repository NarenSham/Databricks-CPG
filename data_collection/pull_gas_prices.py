# data_collection/pull_gas_prices.py
# Pulls weekly gasoline prices from NRCan API
# Saves to data_collection/data/gas_prices.csv

import requests
import zipfile
import io
import pandas as pd

print("Pulling gas prices from NRCan...")

# NRCan gasoline prices — weekly by city/province
api_url = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/18100001/en"
response = requests.get(api_url)
zip_url = response.json()["object"]

zip_response = requests.get(zip_url)
z = zipfile.ZipFile(io.BytesIO(zip_response.content))
csv_file = [f for f in z.namelist() if f.endswith(".csv") and "MetaData" not in f][0]
df = pd.read_csv(z.open(csv_file), low_memory=False)

# Map cities to provinces
CITY_TO_PROVINCE = {
    "Toronto, Ontario": "Ontario",
    "Ottawa-Gatineau, Ontario part, Ontario/Quebec": "Ontario",
    "Thunder Bay, Ontario": "Ontario",
    "Montréal, Quebec": "Quebec",
    "Québec, Quebec": "Quebec",
    "Vancouver, British Columbia": "British Columbia",
    "Victoria, British Columbia": "British Columbia",
    "Edmonton, Alberta": "Alberta",
    "Calgary, Alberta": "Alberta",
    "Winnipeg, Manitoba": "Manitoba",
    "Regina, Saskatchewan": "Saskatchewan",
    "Saskatoon, Saskatchewan": "Saskatchewan"
}

MAJOR_PROVINCES = [
    "Ontario", "Quebec", "British Columbia",
    "Alberta", "Manitoba", "Saskatchewan"
]

# Filter to regular unleaded only
gas = df[df["Type of fuel"].str.contains("Regular unleaded", na=False)].copy()

# Map to province
gas["geo"] = gas["GEO"].map(CITY_TO_PROVINCE)
gas = gas[gas["geo"].isin(MAJOR_PROVINCES)].copy()

# Filter date range
gas["ref_date"] = pd.to_datetime(gas["REF_DATE"])
gas = gas[gas["ref_date"] >= "2017-01-01"].copy()

# Multiple cities per province — average to province level monthly
gas = (gas.groupby(["ref_date", "geo"])["VALUE"]
    .mean()
    .reset_index())

# Add signal metadata
gas["naics_code"] = "457"  # gasoline category
gas["signal_name"] = "gas_price_cents_per_litre"
gas["source"] = "statcan_18100001"
gas["pulled_at"] = pd.Timestamp.now()
gas = gas.rename(columns={"VALUE": "signal_value"})

gas = gas[["ref_date", "geo", "naics_code", 
           "signal_name", "signal_value", "source", "pulled_at"]]

print(f"Filtered rows: {len(gas)}")
print(gas.head(5))

gas.to_csv("data_collection/data/gas_prices.csv", index=False)
print("Saved to data_collection/data/gas_prices.csv")

print(f"Raw rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))