import pandas as pd

df = pd.read_csv("dataset/clean_superstore.csv")

print(df.head())


print(df.dtypes)


features = [

    "Quantity",

    "Discount",

    "Profit"

]

X = df[features]

y = df["Sales"]


print(df[["Sales","Profit","Quantity","Discount"]].corr())

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test=train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

model=LinearRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)

score=r2_score(y_test,prediction)

print("R2 Score :",score)

import joblib

joblib.dump(model,"models/sales_prediction_model_v2.pkl")

print("New Model Saved Successfully!")