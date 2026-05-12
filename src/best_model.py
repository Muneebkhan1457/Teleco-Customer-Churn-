from dotenv import load_dotenv
import os
load_dotenv()

# Explicitly set credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'Muneebkhan1457')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'ab7b436dbc3d1c4caf44d2d81a8f5d27a7e903ad')
os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")
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

    # Get all runs, sorted by ROC-AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"]
    )

    if not runs:
        print("No runs found.")
        return

    # 2. Find the best run that ACTUALY has a model artifact
    best_run = None
    for run in runs:
        run_id = run.info.run_id
        artifacts = client.list_artifacts(run_id)
        artifact_names = [art.path for art in artifacts]
        print(f"Checking Run {run_id[:8]}... Found artifacts: {artifact_names}")
        
        # Check if any artifact looks like a model or the test file
        if "model" in artifact_names or "test.txt" in artifact_names:
            print(f"!!! SUCCESS: Found {artifact_names} in Run {run_id[:8]}")
            best_run = run
            break

    if not best_run:
        print("Could not find any run with a saved model artifact. Please run src/train.py first!")
        return

    best_run_id = best_run.info.run_id
    best_roc_auc = best_run.data.metrics.get("roc_auc", 0)

    print(f"Found Best Run with Model: {best_run_id}")
    print(f"ROC-AUC: {best_roc_auc:.4f}")

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
