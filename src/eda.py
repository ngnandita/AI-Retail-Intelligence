#DataSet ko read kiya#
import pandas as pd
df = pd.read_csv("dataset/SampleSuperstore.csv")
print(df.head())


#Rows#
print(df.shape)
#Columns#
print(df.columns)
#Data Types#
print(df.info())
#Missing Values
print(df.describe())
#Missing Values check#
print(df.isnull().sum())


#Sales by Category Chart#
import matplotlib.pyplot as plt
sales = df.groupby("Category")["Sales"].sum()
sales.plot(kind="bar")
plt.title("Sales by Category")
plt.show()


#Top 10 Products#
top = df.groupby("Sub-Category")["Sales"].sum()
print(top.sort_values(ascending=False).head(10))


#Sales by Region#
region = df.groupby("Region")["Sales"].sum()
print(region)