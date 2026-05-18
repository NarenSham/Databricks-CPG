import requests
import zipfile
import io
import pandas as pd

# Step 1 — get the download URL from StatCan API
# Table 18100004 is CPI
api_url = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/18100004/en"
response = requests.get(api_url)
print(response.status_code)
print(response.json())


# Step 2 — download and extract the zip
zip_url = response.json()["object"]
zip_response = requests.get(zip_url)

# Extract CSV from zip
z = zipfile.ZipFile(io.BytesIO(zip_response.content))
csv_filename = [f for f in z.namelist() if f.endswith(".csv") and "MetaData" not in f][0]
df = pd.read_csv(z.open(csv_filename))

# print(f"Rows: {len(df)}")
# print(f"Columns: {list(df.columns)}")
# print(df.head(3))
# Filter to relevant product groups and provinces
PRODUCT_MAP = {
    "Food purchased from stores": "445",
    "All-items": "455",
    "Health and personal care": "456",
    "Gasoline": "457",
    "Clothing and footwear": "458"
}

PROVINCES = [
    "Ontario", "Quebec", "British Columbia",
    "Alberta", "Manitoba", "Saskatchewan"
]

cpi = df[
    df["Products and product groups"].isin(PRODUCT_MAP.keys()) &
    df["GEO"].isin(PROVINCES)
].copy()

# Map to NAICS codes
cpi["naics_code"] = cpi["Products and product groups"].map(PRODUCT_MAP)

# Clean columns
cpi = cpi[["REF_DATE", "GEO", "naics_code", "VALUE"]].copy()
cpi.columns = ["ref_date", "geo", "naics_code", "cpi_value"]

# Parse date
cpi["ref_date"] = pd.to_datetime(cpi["ref_date"])

# Filter to our date range
cpi = cpi[cpi["ref_date"] >= "2017-01-01"]

# Add metadata
cpi["pulled_at"] = pd.Timestamp.now()

print(f"Rows: {len(cpi)}")
print(cpi.head(5))

# Save locally
cpi.to_csv("data_collection/data/cpi.csv", index=False)
print("Saved to data_collection/data/cpi.csv")
 