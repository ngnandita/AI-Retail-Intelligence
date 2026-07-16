import pandas as pd

import joblib

model = joblib.load("models/sales_prediction_model_v2.pkl")

sample = pd.DataFrame({

    "Quantity":[5],

    "Discount":[0.10],

    "Profit":[120]

})

prediction = model.predict(sample)

print(prediction)