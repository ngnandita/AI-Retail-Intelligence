import pandas as pd
import matplotlib.pyplot as plt
# Load Clean Dataset
df = pd.read_csv("dataset/clean_superstore.csv")
print(df.head())


#Sales by Category#
sales = df.groupby("Category")["Sales"].sum()
print(sales)


#Business chart#
sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()


#Profil by category#
profit = df.groupby("Category")["Profit"].sum()
profit.plot(kind="bar")
plt.title("Profit by Category")
plt.show()


#Sales By Region#
region = df.groupby("Region")["Sales"].sum()
region.plot(kind="pie", autopct="%1.1f%%")
plt.title("Sales by Region")
plt.show()


#Top 10 States#
top_state = df.groupby("State")["Sales"].sum()
top_state = top_state.sort_values(ascending=False).head(10)
top_state.plot(kind="bar")
plt.title("Top 10 States")
plt.show()


#Profit Distribution#
df["Profit"].hist(bins=30)
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()