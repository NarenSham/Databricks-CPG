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

-- Validate
SELECT 'bronze.demand_statcan_retail' AS table_name, COUNT(*) AS row_count
FROM bronze.demand_statcan_retail;