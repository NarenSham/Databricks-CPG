# CPG Compound AI Platform — Model & Architecture Decisions

## Purpose
Documents every significant decision made during development with the 
reasoning behind it. Intended audience: SA interviews, code reviewers, 
future maintainers.

---

## 1. Pooled XGBoost over per-category models

**Decision:** Train one model across all 5 categories and 6 provinces 
rather than 30 separate models.

**Reasoning:**
- 2,940 training rows total. Per-category models would train on ~200 rows 
  each — insufficient for reliable generalization
- Pooled model learns cross-category patterns (seasonality structure is 
  similar across Food, Health, General Merchandise)
- Single champion alias simplifies deployment and governance — one model 
  to register, monitor, and promote
- MAPE 4.68% pooled vs estimated 6-9% for per-category models on this 
  data volume

**Tradeoff accepted:** Clothing (8.6% MAPE) and Gasoline (4.7% MAPE) 
pull the pooled average up. Per-category models might improve these 
specific categories at sufficient data volume. Revisit at 5+ years of 
monthly data or SKU-level grain.

---

## 2. Time-series cross-validation — no random split

**Decision:** Last 12 months as holdout. Never random split.

**Reasoning:**
- Random split on time-series data causes data leakage — future lag 
  features bleed into training
- Last 12 months represents the most recent market conditions, which is 
  exactly what the model will be asked to predict in production
- Holdout period: March 2025 → February 2026

**Tradeoff accepted:** Only one holdout window. Walk-forward validation 
with multiple windows would be more rigorous but requires more data. 
Backtest in notebook 17 partially compensates — 14 months of 
retroactive predictions validate the model on real out-of-sample data.

---

## 3. External signals tested and rejected

**Decision:** Revert to lag-only feature set (10 features) after testing 
CPI, gas prices, and Google Trends.

**Result:** MAPE with signals: 5.83% vs lag-only: 4.68% — signals hurt 
the model.

**Reasoning:**
- lag_1m and lag_12m already capture 98%+ of variance at 
  category/province grain
- External signals add noise, not signal, at this aggregation level
- CPI correlation with target: 0.25. Gas price correlation: 0.18. 
  Both dominated by the lag features

**Revisit trigger:** SKU or brand-level data where macro signals may add 
value that lags cannot capture.

---

## 4. Python tools over UC SQL functions in the agent

**Decision:** Agent in notebook 15 uses Python tool implementations 
rather than calling UC SQL functions registered in notebook 14.

**Reasoning:**
- explain_prediction requires SHAP → business language translation that 
  cannot cleanly live in SQL
- predict_next_month requires MLflow model inference — not possible in 
  a UC SQL function sandbox
- Embedding explanation inline in predict_next_month prevents a second 
  LLM tool call, which UC function chaining cannot do
- UC SQL functions add a network hop with no benefit when the agent 
  already has Spark access

**What notebook 14 is for:** UC functions remain registered as the data 
contract — documentation of what columns exist and what they mean. 
Direct query tools for analysts who don't use the agent. Reference 
implementation for UCFunctionToolkit if it becomes viable.

**MCP path:** When a second agent is built, UC functions become the 
MCP bridge. Each specialist agent registers itself as a UC function 
that the orchestrator discovers via UCFunctionToolkit. This is the 
planned Phase 2 architecture.

---

## 5. InMemorySessionStore now, Lakebase later

**Decision:** Ship with in-memory session storage. Stub the interface 
for Postgres replacement.

**Reasoning:**
- Databricks Free Edition has no Lakebase access at time of build
- InMemorySessionStore and PostgresSessionStore share an identical 
  interface — swap requires zero agent code changes
- Session memory loss on cluster restart is acceptable for the current 
  demo use case

**Trigger to upgrade:** When the Databricks App is built and real 
planners use the system across multiple sessions. Loss of context 
between sessions becomes a real UX problem at that point.

---

## 6. Lazy config loading via _ensure_config_loaded()

**Decision:** CATEGORY_MAPE and CATEGORY_NAMES load from Delta on first 
tool call, not at import time.

**Reasoning:**
- MLflow calls __init__ during model logging validation — no Spark 
  session exists at that moment
- Loading at import time poisoned the global dicts with empty values 
  that persisted into the registered artifact
- Lazy loading with a guard (if CATEGORY_MAPE: return) is safe to call 
  multiple times with no performance cost after first load

**What was fixed:** Removed _ensure_config_loaded() from __init__ after 
diagnosing that the init call during MLflow logging was the root cause 
of "I don't have that data right now" failures.

---

## 7. runpy over dbutils.notebook.run in notebook 18

**Decision:** Pipeline orchestration uses runpy.run_path() to execute 
each notebook file rather than dbutils.notebook.run().

**Reasoning:**
- All pipeline files are .py files that use spark and dbutils from the 
  calling notebook's scope
- runpy.run_path(path, init_globals=globals()) passes the current 
  globals including spark, dbutils, and mlflow to each file
- dbutils.notebook.run() spawns a separate notebook execution context 
  with its own session — cannot share globals

**Tradeoff accepted:** runpy executes in the same process, so a crash 
in one file can affect the calling notebook's state. dbutils.notebook.run() 
is more isolated but requires true Databricks notebook format with 
separate sessions. Acceptable for current scale.

---

## 8. Conditional retraining — 20% MAPE degradation threshold

**Decision:** Only retrain if recent 3-month MAPE exceeds training MAPE 
by more than 20%.

**Reasoning:**
- XGBoost on monthly StatCan data is stable — one new month of data 
  will not meaningfully change the model
- Unnecessary retraining adds latency to the monthly pipeline and risks 
  promoting a worse model if the holdout period is unlucky
- 20% threshold (4.68% → >5.6%) represents genuine degradation rather 
  than normal variance

**Date anchor:** Should_retrain() uses the latest date in 
demand_predictions as the anchor, not current_date(). This prevents 
false negatives when the prediction table hasn't been updated yet for 
the current month.

---

## 9. Merge pattern over overwrite in backtest

**Decision:** notebook 17 uses Delta merge (upsert) keyed on 
ref_date + naics_code + geo rather than overwrite.

**Reasoning:**
- Overwrite destroys historical prediction records on every run
- Merge is idempotent — running twice produces identical results
- Enables incremental monthly updates — only new month is inserted, 
  existing history is untouched
- gold.demand_predictions is the public track record. Overwriting it 
  would corrupt the audit trail

---

## 10. No random split on backtest

**Decision:** Backtest window January 2025 → February 2026. Model was 
trained with holdout through approximately March 2025. Months from 
March 2025 onward are genuinely out-of-sample.

**Honest limitation:** The first ~2 months of the backtest window 
(January–February 2025) overlap with the training period. When 
reporting backtest accuracy publicly, the out-of-sample period is 
March 2025 → February 2026 (12 months), not the full 14.

---

## 11. Agent three-tier eval design

**Tier 1 — Deterministic:** String matching against must_contain and 
must_not_contain lists. Tests guardrails, refusals, technical term 
suppression. No LLM needed.

**Tier 1b — Forecast range:** Extracts dollar value from response via 
regex. Checks against expected range with ±15% tolerance. The only tier 
that validates XGBoost model correctness end-to-end.

**Tier 2 — Citation grounded:** Required concepts must be present, 
forbidden technical terms must be absent. Validates SHAP translation 
and data grounding.

**Tier 3 — LLM judged:** mlflow.genai.evaluate with RelevanceToQuery 
and Safety scorers. Tests reasoning quality on questions with no single 
right answer. Currently enabled — run after each champion promotion.

**Known false negatives:** Some Tier 1 and 2 checks are sensitive to 
LLM phrasing variation. Test cases updated to match actual agent 
language rather than expected language where the agent's phrasing is 
correct but different.

---

## 12. Multi-agent architecture — future state

**Decision:** Build specialist agents (demand, price, promo) as fully 
independent systems. Connect via orchestrator using MCP.

**Pattern:** Each specialist is a complete autonomous agent with its own 
tools, Delta tables, system prompt, eval harness, and MLflow 
registration. The orchestrator is an MCP client that discovers 
specialists via UCFunctionToolkit and routes questions to the right 
domain.

**Why not shared tool registry:** Domain knowledge (Clothing has low 
confidence, seasonality drives Food) belongs inside each specialist 
agent, not in a shared tool layer. Independent evals and independent 
deployment are only possible if agents are genuinely separate.

**Build order:** Demand agent complete. App next. Price agent after app 
ships. Orchestrator after both specialists are stable and evaluated.

---

## Data decisions

**Territories excluded:** Yukon, NWT, Nunavut dropped due to 65%+ null 
suppression rate. Not representative retail markets.

**NAICS code updates:** StatCan updated codes between data vintages. 
Mapped 446→456, 447→457, 448→458, 452→455. Code 454 not present in 
data.

**6 provinces only:** Ontario, Quebec, BC, Alberta, Manitoba, 
Saskatchewan. Represent 95%+ of Canadian retail sales.

**Google Trends broadcast:** Canada-level trends signal cross-joined to 
all 6 provinces in silver. Provincial-level trends not available via 
pytrends at monthly grain.

**Weather signal deferred:** No clean monthly provincial weather source 
found. Deferred to future iteration. Month feature partially captures 
seasonal weather effects already.