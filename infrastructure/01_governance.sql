-- 01_governance.sql
-- Shared audit tables used by all agents. Run after 00_catalog_and_schemas.sql.

USE CATALOG cpg_planning;
USE SCHEMA governance;

-- Every agent decision logged here
CREATE TABLE IF NOT EXISTS ai_decision_log (
  decision_id    STRING,
  logged_at      TIMESTAMP,
  agent_name     STRING,
  action         STRING,
  details        STRING
);

-- Every tool call logged here
CREATE TABLE IF NOT EXISTS tool_call_history (
  call_id        STRING,
  called_at      TIMESTAMP,
  agent_name     STRING,
  tool_name      STRING,
  parameters     STRING,
  result_summary STRING
);

-- Validate
SELECT table_name
FROM system.information_schema.tables
WHERE table_catalog = 'cpg_planning'
AND table_schema = 'governance'
ORDER BY table_name;