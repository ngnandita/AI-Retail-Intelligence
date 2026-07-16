import pandas as pd
df = pd.read_csv("dataset/clean_superstore.csv")
print(df.head())


features = ["Quantity", "Discount"]
X = df[features]
y = df["Sales"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

#Prediction#
prediction = model.predict(X_test)
print(prediction[:10])

#Accuracy CHeck
from sklearn.metrics import r2_score
score = r2_score(y_test, prediction)
print("R2 Score :", score)


#Save model#
import joblib
joblib.dump(model, "models/sales_prediction_model.pkl")
print("Model Saved Successfully!")