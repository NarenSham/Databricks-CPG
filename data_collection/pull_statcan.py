# data_collection/pull_statcan.py

import requests
import zipfile
import io
import pandas as pd
import re
from pathlib import Path

OUTPUT_DIR = Path("data_collection/data")


# ── Fetch table ─────────────────────────────────────────
def fetch_table(table_id):
    print(f"\nFetching table {table_id}...")

    meta = requests.get(
        f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{table_id}/en",
        timeout=30
    ).json()

    r = requests.get(meta["object"], timeout=120)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = [n for n in z.namelist()
                    if n.endswith(".csv") and "MetaData" not in n][0]
        df = pd.read_csv(z.open(csv_name), encoding="utf-8")

    print(f"  Raw rows: {len(df)}")
    return df


# ── Find NAICS column ───────────────────────────────────
def find_naics_column(df):
    for col in df.columns:
        if "NAICS" in col:
            return col
    raise ValueError("NAICS column not found")


# ── NAICS extraction (robust) ───────────────────────────
def extract_naics(desc):
    if pd.isna(desc):
        return None

    text = str(desc)

    # Handle "44-45"
    if "44-45" in text:
        return "44"

    # Extract from brackets
    match = re.search(r"\[(\d{3,5})\]", text)
    if match:
        return match.group(1)

    # Fallback: leading digits
    match = re.match(r"(\d{3,5})", text)
    if match:
        return match.group(1)

    return None


# ── Province filter ─────────────────────────────────────
def is_province(geo):
    geo = str(geo)

    if geo == "Canada":
        return False

    if "," in geo:  # removes cities like "Toronto, Ontario"
        return False

    return True


# ── CORE: keep deepest NAICS per branch ─────────────────
def filter_leaf_level(df):
    df = df.copy()

    df["naics_len"] = df["naics_code"].str.len()
    df["naics_3"] = df["naics_code"].str[:3]

    # Keep deepest level within each NAICS 3 branch
    df = (
        df.sort_values("naics_len")
        .groupby(["geo", "ref_date", "naics_3"], as_index=False)
        .tail(1)
    )

    return df


# ── Main pipeline ───────────────────────────────────────
def pull():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── TABLE 1 ─────────────────────────────────────────
    df1 = fetch_table("20100056")
    naics_col1 = find_naics_column(df1)

    print("\nSample GEO values:")
    print(df1["GEO"].unique()[:10])

    print("\nSample NAICS values:")
    print(df1[naics_col1].dropna().unique()[:10])

    df1 = df1[df1["GEO"].apply(is_province)].copy()
    df1["naics_code"] = df1[naics_col1].apply(extract_naics)
    df1["source_table"] = "20100056"

    # ── TABLE 2 (optional) ──────────────────────────────
    try:
        df2 = fetch_table("20100057")
        df2 = df2[df2["GEO"].apply(is_province)].copy()

        # No NAICS → assign 454
        df2["naics_code"] = "454"
        df2["source_table"] = "20100057"

        combined = pd.concat([df1, df2], ignore_index=True)

    except Exception as e:
        print(f"  Table 20100057 issue ({e}) — skipping")
        combined = df1

    # ── Standardize ─────────────────────────────────────
    combined["ref_date"] = pd.to_datetime(combined["REF_DATE"])
    combined["pulled_at"] = pd.Timestamp.now()

    combined = combined.rename(columns={
        "GEO": "geo",
        naics_col1: "naics_description",
        "VALUE": "value",
        "STATUS": "status",
    })

    combined = combined[combined["naics_code"].notna()]

    combined = combined[[
        "ref_date",
        "geo",
        "naics_code",
        "naics_description",
        "value",
        "status",
        "source_table",
        "pulled_at"
    ]]

    # ── BACKBONE (NAICS 3) ──────────────────────────────
    df_naics3 = (
        combined
        .assign(naics_3=combined["naics_code"].str[:3])
        .groupby(["geo", "ref_date", "naics_3"], as_index=False)
        ["value"].sum()
    )

    # ── LEAF LEVEL (deepest per branch) ─────────────────
    df_leaf = filter_leaf_level(combined)

    # ── Save outputs ────────────────────────────────────
    combined.to_csv(OUTPUT_DIR / "statcan_full.csv", index=False)
    df_naics3.to_csv(OUTPUT_DIR / "statcan_naics3.csv", index=False)
    df_leaf.to_csv(OUTPUT_DIR / "statcan_leaf_level.csv", index=False)

    # ── Diagnostics ─────────────────────────────────────
    print("\nRows per province (leaf level):")
    print(df_leaf.groupby("geo")["value"].count().to_string())

    print("\nUnique NAICS (leaf level):")
    print(df_leaf["naics_code"].nunique())

    print(f"\nDate range: {df_leaf['ref_date'].min().date()} → {df_leaf['ref_date'].max().date()}")

    print(f"\nSaved:")
    print(f"  → statcan_full.csv")
    print(f"  → statcan_naics3.csv")
    print(f"  → statcan_leaf_level.csv")


# ── Entry point ─────────────────────────────────────────
if __name__ == "__main__":
    pull()