# data_collection/pull_google_trends.py
import pandas as pd
import time
from pytrends.request import TrendReq

print("Pulling Google Trends data...")

pytrends = TrendReq(hl="en-CA", tz=-300)

# Search terms mapped to NAICS categories
TERMS = {
    "445": ["grocery delivery", "food prices canada"],
    "455": ["walmart canada", "costco canada"],
    "456": ["pharmacy canada", "shoppers drug mart"],
    "457": ["gas prices canada", "fuel prices"],
    "458": ["clothing sale canada", "fashion canada"]
}

all_records = []

for naics, keywords in TERMS.items():
    for keyword in keywords:
        try:
            pytrends.build_payload(
                [keyword],
                geo="CA",
                timeframe="2017-01-01 2026-02-28"
            )
            df = pytrends.interest_over_time()
            
            if df.empty:
                print(f"No data for: {keyword}")
                continue
                
            df = df.reset_index()
            df["naics_code"] = naics
            df["signal_name"] = f"trends_{keyword.replace(' ', '_')}"
            df["geo"] = "Canada"
            df["source"] = "google_trends"
            df["pulled_at"] = pd.Timestamp.now()
            df = df.rename(columns={keyword: "signal_value", "date": "ref_date"})
            df = df[["ref_date", "geo", "naics_code",
                     "signal_name", "signal_value", "source", "pulled_at"]]
            all_records.append(df)
            print(f"Pulled: {keyword} ({len(df)} rows)")
            time.sleep(2)  # avoid rate limiting
            
        except Exception as e:
            print(f"Failed: {keyword} — {e}")
            time.sleep(5)

if all_records:
    trends = pd.concat(all_records, ignore_index=True)
    print(f"\nTotal rows: {len(trends)}")
    print(trends.head(5))
    trends.to_csv("data_collection/data/google_trends.csv", index=False)
    print("Saved to data_collection/data/google_trends.csv")
else:
    print("No data pulled")