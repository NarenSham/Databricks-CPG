-- pipeline/demand_sensing_pipeline.sql
-- Declarative Pipeline for demand sensing data layer
-- Handles bronze ingestion and silver cleaning with quality gates
-- Replaces manual execution of 01_load_statcan_to_bronze and 03_bronze_to_silver

-- ============================================================
-- BRONZE: Raw ingestion with quality gates
-- ============================================================

CREATE OR REFRESH STREAMING TABLE bronze_statcan_retail_raw
COMMENT 'Raw StatCan retail data as landed from Volume landing zone'
AS SELECT * FROM STREAM read_files(
  '/Volumes/cpg_planning/bronze/landing/',
  format => 'csv',
  header => 'true',
  inferSchema => 'true',
  pathGlobFilter => 'statcan_retail.csv'
);

CREATE OR REFRESH STREAMING TABLE bronze_statcan_retail_clean (
  CONSTRAINT valid_date
    EXPECT (ref_date IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT no_future_dates
    EXPECT (ref_date <= current_date())
    ON VIOLATION FAIL UPDATE,
  CONSTRAINT positive_or_suppressed
    EXPECT (value > 0 OR status IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT known_geography
    EXPECT (geo IN (
      'Ontario','Quebec','British Columbia',
      'Alberta','Manitoba','Saskatchewan',
      'Newfoundland and Labrador','New Brunswick',
      'Nova Scotia','Prince Edward Island',
      'Northwest Territories','Nunavut','Yukon'
    ))
    ON VIOLATION DROP ROW
)
COMMENT 'Quality-gated StatCan retail data. Rows failing constraints are dropped or pipeline halts.'
AS SELECT
  CAST(ref_date AS DATE)       AS ref_date,
  geo,
  CAST(naics_code AS STRING)   AS naics_code,
  naics_description,
  CAST(value AS DOUBLE)        AS value,
  status,
  source_table,
  CAST(pulled_at AS TIMESTAMP) AS pulled_at
FROM STREAM(LIVE.bronze_statcan_retail_raw);

-- ============================================================
-- SILVER: Clean, filter to target categories and provinces
-- ============================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_retail_monthly (
  CONSTRAINT target_categories
    EXPECT (naics_code IN ('445','455','456','457','458'))
    ON VIOLATION DROP ROW,
  CONSTRAINT major_provinces
    EXPECT (geo IN (
      'Ontario','Quebec','British Columbia',
      'Alberta','Manitoba','Saskatchewan'
    ))
    ON VIOLATION DROP ROW,
  CONSTRAINT clean_status
    EXPECT (status NOT IN ('x','F','..') OR status IS NULL)
    ON VIOLATION DROP ROW
)
COMMENT 'Clean monthly retail trade. 5 categories, 6 major provinces, suppressed values removed.'
AS SELECT
  ref_date,
  geo,
  naics_code,
  naics_description,
  value,
  status
FROM LIVE.bronze_statcan_retail_clean;