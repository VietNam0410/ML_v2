import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Preprocessing import preprocess
from utils.config import setup_mlflow

def train_and_log():
    """ Load dữ liệu, train mô hình và log vào MLflow """
    
    setup_mlflow()  # Cấu hình MLflow

    df = preprocess.load_data("Titanic_Liner_Regression/Data/titanic.csv")
    X_train, X_test, y_train, y_test = preprocess.preprocess_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"✅ Model trained with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_log()
