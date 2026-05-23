import mlflow
import sys
from mlflow.tracking import MlflowClient
# Add the workspace notebooks directory to the Python path
notebooks_dir = f'/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/Databricks-CPG/notebooks'
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

from Utils.governance_logging import log_decision

username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{username}/Databricks-CPG/experiments/demand_sensing"

client = MlflowClient()

# Get best run by lowest MAPE
experiment = client.get_experiment_by_name(experiment_path)
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName LIKE 'xgboost%'",
    order_by=["metrics.mape ASC"],
    max_results=1
)[0]

print(f"Best run: {best_run.info.run_id}")
print(f"Best MAPE: {best_run.data.metrics['mape']:.4f}")

# Register to Unity Catalog
registered = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name="cpg_planning.ml.demand_model"
)

# Set as champion
client.set_registered_model_alias(
    name="cpg_planning.ml.demand_model",
    alias="champion",
    version=registered.version
)

print(f"Registered version: {registered.version}")
print(f"Champion alias set on: cpg_planning.ml.demand_model")

log_decision(
    agent_name="demand_agent",
    action="model_promoted_to_champion",
    details=f"Model version {registered.version} promoted to champion alias. MAPE: {best_run.data.metrics['mape']:.4f}. Run ID: {best_run.info.run_id}.",
    model_version=str(registered.version)
)
