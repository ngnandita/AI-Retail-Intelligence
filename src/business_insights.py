import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/clean_superstore.csv")


#Top 10 Prodiucts#
import matplotlib.pyplot as plt

top_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

# Long names short karo
top_products.index = top_products.index.str[:30] + "..."
plt.figure(figsize=(14,8))
top_products.plot(kind="bar")

plt.title("Top 10 Products by Sales", fontsize=16)
plt.xlabel("Products")
plt.ylabel("Sales")
plt.xticks(rotation=70, ha="right", fontsize=8)
plt.tight_layout()
plt.show()


#Top Customers#
top_customer = (
    df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customer)

top_customer.plot(kind="bar")
plt.title("Top Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Highest Profit State#
profit_state = (
    df.groupby("State")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(profit_state.head(10))

profit_state.head(10).plot(kind="bar")
plt.title("Top Profit States")
plt.tight_layout()
plt.show()


#Loss Making States#
loss_state = profit_state.tail(10)
print(loss_state)

loss_state.plot(kind="bar")
plt.title("Loss Making States")
plt.tight_layout()
plt.show()


#Average Discount#
discount = df.groupby("Category")["Discount"].mean()
print(discount)


#Correlation#
print(df[["Sales","Profit","Discount","Quantity"]].corr())