import mlflow
import mlflow.sklearn

def load_model():
    """ Load mô hình đã được lưu trong MLflow """
    model = mlflow.sklearn.load_model("mlruns/0/random_forest_model")
    return model

if __name__ == "__main__":
    model = load_model()
    print("🎯 Model loaded successfully!")
