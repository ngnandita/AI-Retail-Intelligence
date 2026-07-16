import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

df = pd.read_csv("dataset/clean_superstore.csv")

features = [
    "Quantity",
    "Discount",
    "Profit"
]
X = df[features]
y = df["Sales"]


model = XGBRegressor(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=6,

    random_state=42

)

model.fit(X, y)

importance = model.feature_importances_

for feature, score in zip(features, importance):

    print(feature, ":", score)

    plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("charts/feature_importance.png")

plt.show()

import joblib

joblib.dump(

    model,

    "models/xgboost_best.pkl"

)

print("Improved Model Saved")