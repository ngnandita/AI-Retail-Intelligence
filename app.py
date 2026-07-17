import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os


# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="AI Retail Intelligence",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    color:#1565C0;
}

h2{
    color:#00897B;
}

div[data-testid="metric-container"]{
    background:#ffffff;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

header{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛒 AI Retail Intelligence")

st.sidebar.markdown("---")

st.sidebar.info("""
👩‍💻 Developer

Nandita Gargayan

B.Tech CSE

LNCT Group of Colleges
""")

st.sidebar.markdown("---")




uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("dataset/clean_superstore.csv")

st.sidebar.header("Filters")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

if category != "All":
    filtered_df = df[df["Category"] == category]
else:
    filtered_df = df

    st.sidebar.download_button(
    "📥 Download Dataset",
    df.to_csv(index=False),
    file_name="superstore_dataset.csv",
    mime="text/csv"
)

# ============================
# Load ML Model
# ============================
model = joblib.load("models/xgboost_model.pkl")

st.markdown("---")
st.header("Prediction History")

history = pd.read_csv("history/prediction_history.csv")

st.dataframe(
    history,
    use_container_width=True
)

search = st.text_input("🔍 Search by Quantity")

if search:
    result = history[
        history["Quantity"].astype(str).str.contains(search)
    ]
    st.dataframe(
        result,
        use_container_width=True
    )

    st.download_button(
    "📥 Download Prediction History",
    history.to_csv(index=False),
    file_name="prediction_history.csv",
    mime="text/csv"
)


# ============================
# Title
# ============================

st.markdown("""
# 🛒 AI Retail Intelligence

### AI Powered Business Analytics Dashboard

Predict Sales • Business Insights • Machine Learning • Data Analytics
""")

st.markdown("---")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "💰 Total Sales",
    f"₹ {df['Sales'].sum():,.0f}"
)

kpi2.metric(
    "📈 Total Profit",
    f"₹ {df['Profit'].sum():,.0f}"
)

kpi3.metric(
    "📦 Orders",
    len(df)
)

kpi4.metric(
    "🎯 Avg Discount",
    f"{df['Discount'].mean()*100:.2f}%"
)


# ============================
# Metrics
# ============================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"{df['Sales'].sum():,.2f}"
)

col2.metric(
    "Total Profit",
    f"{df['Profit'].sum():,.2f}"
)

col3.metric(
    "Orders",
    len(df)
)

st.markdown("---")

st.success(
    "✅ Welcome to AI Retail Intelligence Dashboard"
)

# ============================
# Sales Prediction
# ============================

st.header("Sales Prediction")

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=5
)

discount = st.number_input(
    "Discount",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

profit = st.number_input(
    "Profit",
    value=120.0
)
if st.button("Predict Sales"):

    sample = pd.DataFrame({

        "Quantity":[quantity],

        "Discount":[discount],

        "Profit":[profit]

    })

    prediction = model.predict(sample)

    history_path = "history/prediction_history.csv"

history = pd.DataFrame({
    "Quantity":[quantity],
    "Discount":[discount],
    "Profit":[profit],
    "Predicted_Sales":[prediction[0]]
})

history.to_csv(
    history_path,
    mode="a",
    header=not os.path.exists(history_path),
    index=False
)

    st.success(
        f"Predicted Sales : ₹ {prediction[0]:.2f}"
    )

    prediction_text = f"""
Quantity : {quantity}

Discount : {discount}

Profit : {profit}

Predicted Sales : {prediction[0]:.2f}
"""

    st.download_button(
        "Download Prediction",
        prediction_text,
        file_name="prediction.txt"
    )

    st.balloons()

# ============================
# Business Dashboard
# ============================

st.header("Business Dashboard")
pie = px.pie(

    df,

    names="Category",

    values="Sales",

    title="Category Distribution"

)

st.plotly_chart(
    pie,
    use_container_width=True
)

st.download_button(
    label="Download Dashboard Data",
    data=filtered_df.to_csv(index=False),
    file_name="dashboard_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.header("Monthly Sales Trend")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)

monthly_sales = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)

monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)

fig3 = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)
st.plotly_chart(
    fig3,
    use_container_width=True
)

#Show Dataset Preview#
st.markdown("---")
st.header("Dataset Preview")

#DataSet Information#
st.header("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", filtered_df.shape[0])
with col2:
    st.metric("Columns", filtered_df.shape[1])


st.dataframe(
    filtered_df.head(10),
    use_container_width=True
)

# Sales by Category

st.subheader("Sales by Category")



category_sales = (
    filtered_df.groupby("Category")["Sales"]
      .sum()
      .reset_index()
)

fig = px.bar(

    category_sales,

    x="Category",

    y="Sales",

    color="Category",

    title="Sales by Category"

)


# Top Products

st.subheader("Top 10 Sub-Categories")


top_products = (
    df.groupby("Sub-Category")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

fig2 = px.bar(

    top_products,

    x="Sub-Category",

    y="Sales",

    color="Sales",

    title="Top 10 Products"

)
st.plotly_chart(
    fig2,
    use_container_width=True
)



st.markdown("---")

st.markdown("---")
st.header("Region Wise Sales")

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig_region = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    color="Region",
    title="Sales by Region"
)

st.plotly_chart(
    fig_region,
    use_container_width=True,
    key="region_sales"
)

st.header("Segment Wise Sales")

segment_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .reset_index()
)

fig_segment = px.pie(
    segment_sales,
    names="Segment",
    values="Sales",
    hole=0.4,
    title="Segment Distribution"
)

st.plotly_chart(
    fig_segment,
    use_container_width=True,
    key="segment_chart"
)

# ============================
# Business Insights
# ============================

st.header("Business Insights")

best_category = (
    df.groupby("Category")["Sales"]
      .sum()
      .idxmax()
)

st.success(
    f"Highest Sales Category : {best_category}"
)

st.info(
    f"Total Sales : ₹ {df['Sales'].sum():,.2f}"
)

st.info(
    f"Total Profit : ₹ {df['Profit'].sum():,.2f}"
)

st.info(
    f"Average Discount : {df['Discount'].mean():.2%}"
)






st.markdown("---")
st.header("Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "R2 Score": [
        0.32,
        0.36,
        0.39
    ]
})

st.dataframe(performance, use_container_width=True)

best_model = performance.loc[
    performance["R2 Score"].idxmax(),
    "Model"
]

st.success(f"🏆 Best Performing Model: {best_model}")

st.markdown("---")

st.markdown(
    """
    ### 👩‍💻 Developed by Nandita Gargayan

    AI Retail Intelligence Project

    Built using Python, Streamlit, XGBoost and Data Analytics.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="sales_chart"
    )

with col2:
    st.plotly_chart(
        pie,
        use_container_width=True,
        key="pie_chart"
    )

    best_month = monthly_sales.loc[
    monthly_sales["Sales"].idxmax()
]

st.success(
    f"🏆 Best Sales Month: {best_month['Order Date']} (₹ {best_month['Sales']:,.2f})"
)

st.markdown("---")

st.header("Executive Summary")

st.write(f"""
• Total Orders : **{len(df)}**

• Total Sales : **₹ {df['Sales'].sum():,.2f}**

• Total Profit : **₹ {df['Profit'].sum():,.2f}**

• Best Category : **{best_category}**

• Average Discount : **{df['Discount'].mean()*100:.2f}%**
""")

##Correlation Heatmap#
st.markdown("---")
st.header("Correlation Analysis")
corr = filtered_df.select_dtypes(include="number").corr()
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(corr)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
st.pyplot(fig)

#Footer#
st.markdown("---")

st.markdown("""
<center>

### 👩‍💻 Developed by Nandita Gargayan

AI Retail Intelligence Dashboard

Python • Streamlit • XGBoost • Plotly

© 2026 All Rights Reserved

</center>
""", unsafe_allow_html=True)

#AI Business Recommendation#
st.markdown("---")
st.header("🤖 AI Business Recommendation")

best_category = df.groupby("Category")["Profit"].sum().idxmax()
worst_category = df.groupby("Category")["Profit"].sum().idxmin()

st.success(
    f"✅ Highest Profit Category : {best_category}"
)

st.warning(
    f"⚠️ Lowest Profit Category : {worst_category}"
)

st.info(
    "Recommendation: Increase inventory and marketing for the highest-profit category while improving pricing or cost strategy for the lowest-profit category."
)

if st.button("🗑 Clear Prediction History"):

    pd.DataFrame(
        columns=[
            "Quantity",
            "Discount",
            "Profit",
            "Predicted_Sales"
        ]
    ).to_csv(
        "history/prediction_history.csv",
        index=False
    )

    st.success("History Cleared Successfully")