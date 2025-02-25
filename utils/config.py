import mlflow

def setup_mlflow():
    """ Cấu hình MLflow tracking URI """
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Titanic Experiment")
