-- 00_catalog_and_schemas.sql
-- Run once to bootstrap the platform. Safe to re-run.

CREATE CATALOG IF NOT EXISTS cpg_planning;

CREATE SCHEMA IF NOT EXISTS cpg_planning.bronze;
CREATE SCHEMA IF NOT EXISTS cpg_planning.silver;
CREATE SCHEMA IF NOT EXISTS cpg_planning.gold;
CREATE SCHEMA IF NOT EXISTS cpg_planning.governance;
CREATE SCHEMA IF NOT EXISTS cpg_planning.ml;
CREATE SCHEMA IF NOT EXISTS cpg_planning.monitoring;

CREATE VOLUME IF NOT EXISTS cpg_planning.bronze.landing;

-- Validate
SELECT schema_name
FROM system.information_schema.schemata
WHERE catalog_name = 'cpg_planning'
ORDER BY schema_name;