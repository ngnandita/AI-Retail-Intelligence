import pandas as pd
import plotly.express as px

# Load Dataset
df = pd.read_csv("dataset/clean_superstore.csv")

# Sales by Category
sales = df.groupby("Category")["Sales"].sum().reset_index()

# Interactive Chart
fig = px.bar(
    sales,
    x="Category",
    y="Sales",
    title="Sales by Category",
    color="Category",
    text_auto=True
)

fig.write_html("dashboard/retail_dashboard.html")

fig.show()


#Sales by Region#
region = df.groupby("Region")["Sales"].sum().reset_index()
fig = px.pie(
    region,
    names="Region",
    values="Sales",
    title="Sales by Region"
)
fig.show()

#Profit by catgory#
profit = df.groupby("Category")["Profit"].sum().reset_index()
fig = px.bar(
    profit,
    x="Category",
    y="Profit",
    color="Category",
    title="Profit by Category",
    text_auto=True
)
fig.show()

#Top 10 States#
state = (
    df.groupby("State")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

fig = px.bar(
    state,
    x="State",
    y="Sales",
    title="Top 10 States"
)

fig.show()