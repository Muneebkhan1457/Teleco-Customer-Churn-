from dotenv import load_dotenv
import os
load_dotenv()

# Explicitly set credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='Muneebkhan1457', repo_name='Teleco-Customer-Churn-', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Muneebkhan1457/Teleco-Customer-Churn-.mlflow")

def register_best_model():
    client = MlflowClient()
    
    # 1. Search for runs in this experiment
    experiment = client.get_experiment_by_name("Telco-Churn-Prediction")
    if not experiment:
        print("Experiment not found!")
        return

    # 2. Get the latest runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"] # Get LATEST runs first
    )

    if not runs:
        print("No runs found.")
        return

    # 3. Find the best run from the RECENT ones
    best_run = None
    max_roc_auc = -1
    
    for run in runs[:5]: # Check the 5 most recent runs
        run_id = run.info.run_id
        roc_auc = run.data.metrics.get("roc_auc", 0)
        
        artifacts = client.list_artifacts(run_id)
        artifact_names = [art.path for art in artifacts]
        print(f"Checking Run {run_id[:8]} | ROC-AUC: {roc_auc:.4f} | Artifacts: {artifact_names}")
        
        if "model" in artifact_names:
            if roc_auc >= max_roc_auc:
                max_roc_auc = roc_auc
                best_run = run

    if not best_run:
        print("Could not find a recent run with a 'model' artifact.")
        return

    best_run_id = best_run.info.run_id
    print(f"✅ Selected Best Recent Run: {best_run_id} with ROC-AUC: {max_roc_auc:.4f}")

    # 3. Register this model
    model_name = "Telco-Churn-Model"
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        print(f"Registering model from run {best_run_id}...")
        result = mlflow.register_model(model_uri, model_name)
        print(f"Successfully registered model '{model_name}' (Version {result.version})")
    except Exception as e:
        print(f"Error registering model: {e}")

if __name__ == "__main__":
    register_best_model()
