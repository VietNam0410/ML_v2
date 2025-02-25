import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="Titanic_Liner_Regression/Data/titanic.csv"):
    """ Load dữ liệu từ file CSV """
    return pd.read_csv(path)

def preprocess_data(df):
    """ Tiền xử lý dữ liệu: điền giá trị thiếu, mã hóa, chuẩn hóa """
    df = df.drop(columns=["Name", "Ticket", "Cabin"], errors='ignore')

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
