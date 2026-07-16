import pandas as pd
import joblib

model = joblib.load("models/xgboost_model.pkl")

sample = pd.DataFrame({

    "Quantity":[5],

    "Discount":[0.10],

    "Profit":[120]

})

prediction = model.predict(sample)

print("Predicted Sales:", prediction[0])