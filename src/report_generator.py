import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/clean_superstore.csv")

# Business KPIs
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = len(df)
average_sales = df["Sales"].mean()

top_category = df.groupby("Category")["Sales"].sum().idxmax()
top_state = df.groupby("State")["Profit"].sum().idxmax()
top_customer = df.groupby("Customer Name")["Sales"].sum().idxmax()

# Generate Report
report = f"""
=============================
AI RETAIL INTELLIGENCE REPORT
=============================

Total Orders : {total_orders}

Total Sales : {total_sales:.2f}

Total Profit : {total_profit:.2f}

Average Sales : {average_sales:.2f}

Top Category : {top_category}

Top Profit State : {top_state}

Top Customer : {top_customer}

Recommendations

1. Increase stock of top-selling category.
2. Reward loyal customers.
3. Reduce discounts where profit is low.
4. Focus marketing on profitable states.

=============================
"""

# Save Report
with open("reports/business_report.txt", "w") as file:
    file.write(report)

print("Business Report Generated Successfully!")