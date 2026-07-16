import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)
from xgboost import XGBRegressor

df = pd.read_csv("dataset/clean_superstore.csv")

features = [
    "Quantity",
    "Discount",
    "Profit"
]

X = df[features]

y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

model = XGBRegressor(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=5,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

prediction = model.predict(X_test)

print("XGBoost Results")

print("R2 Score :", r2_score(y_test, prediction))

print("MAE :", mean_absolute_error(y_test, prediction))

print("RMSE :", root_mean_squared_error(y_test, prediction))

# Save the trained model
joblib.dump(model, "models/xgboost_model.pkl")

print("XGBoost Model Saved Successfully!")