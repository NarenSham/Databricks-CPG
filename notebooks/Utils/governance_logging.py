# notebooks/01_demand_sensing/12_governance_logging.py
# Governance logging utility
# Import this in any notebook to log decisions to the audit table

import uuid
from datetime import datetime, timezone
from pyspark.sql import Row, SparkSession

def log_decision(
    agent_name: str,
    action: str,
    details: str,
    model_version: str = None
):
    """
    Log a decision to the governance audit table.
    Call this from any notebook for any significant event.
    """
    # Get the active SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    row = Row(
        decision_id=str(uuid.uuid4()),
        logged_at=datetime.now(timezone.utc),
        agent_name=agent_name,
        action=action,
        details=details
    )
    
    df = spark.createDataFrame([row])
    
    (df.write
        .format("delta")
        .mode("append")
        .saveAsTable("cpg_planning.governance.ai_decision_log"))

def log_tool_call(
    agent_name: str,
    tool_name: str,
    parameters: str,
    result_summary: str
):
    """
    Log a tool call to the tool call history table.
    """
    # Get the active SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    row = Row(
        call_id=str(uuid.uuid4()),
        called_at=datetime.now(timezone.utc),
        agent_name=agent_name,
        tool_name=tool_name,
        parameters=parameters,
        result_summary=result_summary
    )
    
    df = spark.createDataFrame([row])
    
    (df.write
        .format("delta")
        .mode("append")
        .saveAsTable("cpg_planning.governance.tool_call_history"))
