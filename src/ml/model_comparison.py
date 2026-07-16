import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

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

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

rf = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

print("\nLinear Regression")

print("R2 Score :", r2_score(y_test, lr_prediction))

print("MAE :", mean_absolute_error(y_test, lr_prediction))

print("RMSE :", root_mean_squared_error(y_test, lr_prediction))

print("\nRandom Forest")

print("R2 Score :", r2_score(y_test, rf_prediction))

print("MAE :", mean_absolute_error(y_test, rf_prediction))

print("RMSE :", root_mean_squared_error(y_test, rf_prediction))


joblib.dump(

    rf,

    "models/random_forest_model.pkl"

)

print("Random Forest Saved Successfully!")