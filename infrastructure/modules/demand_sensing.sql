-- infrastructure/modules/demand_sensing.sql
-- Demand module tables: bronze, silver, gold.
-- Run after 00_catalog_and_schemas.sql and 01_governance.sql.

USE CATALOG cpg_planning;

-- ============================================================
-- BRONZE: Source-faithful. Store exactly what StatCan provides.
-- ============================================================

CREATE TABLE IF NOT EXISTS bronze.demand_statcan_retail (
  ref_date          DATE,
  geo               STRING,
  naics_code        STRING,
  naics_description STRING,
  value             DOUBLE,
  status            STRING,
  source_table      STRING,
  pulled_at         TIMESTAMP
);


-- Bronze: all external signals in one table
CREATE TABLE IF NOT EXISTS bronze.demand_signals (
  ref_date          DATE,
  geo               STRING,
  naics_code        STRING,
  signal_name       STRING,
  signal_value      DOUBLE,
  source            STRING,
  pulled_at         TIMESTAMP
);


-- ============================================================
-- Silver: cleaned retail trade
-- ============================================================

CREATE TABLE IF NOT EXISTS silver.demand_retail_monthly (
  ref_date          DATE,
  geo               STRING,
  naics_code        STRING,
  naics_description STRING,
  value             DOUBLE,
  status            STRING
);

-- ============================================================
-- Gold: feature table
-- ============================================================

CREATE TABLE IF NOT EXISTS gold.demand_feature_table (
  ref_date          DATE,
  geo               STRING,
  naics_code        STRING,
  naics_description STRING,
  value             DOUBLE,
  status            STRING,
  lag_1m            DOUBLE,
  lag_2m            DOUBLE,
  lag_3m            DOUBLE,
  lag_6m            DOUBLE,
  lag_12m           DOUBLE,
  rolling_3m_avg    DOUBLE,
  rolling_6m_avg    DOUBLE,
  month             INTEGER,
  naics_encoded     INTEGER,
  geo_encoded       INTEGER
);

-- Gold: model predictions
CREATE TABLE IF NOT EXISTS gold.demand_predictions (
  prediction_id     STRING,
  naics_code        STRING,
  geo               STRING,
  target_month      DATE,
  predicted_value   DOUBLE,
  actual_value      DOUBLE,
  model_version     STRING,
  prediction_date   TIMESTAMP,
  mape              DOUBLE
);

-- Validate
SELECT 'bronze.demand_statcan_retail' AS table_name, COUNT(*) AS row_count
FROM bronze.demand_statcan_retail;

SELECT table_name
FROM system.information_schema.tables
WHERE table_catalog = 'cpg_planning'
AND table_schema IN ('bronze', 'silver', 'gold')
ORDER BY table_schema, table_name;