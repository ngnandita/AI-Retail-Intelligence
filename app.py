import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="AI Retail Intelligence",
    page_icon="🛒",
    layout="wide"
)

st.sidebar.title("🛒 AI Retail Intelligence")




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

# ============================
# Load ML Model
# ============================

model = joblib.load("models/xgboost_model.pkl")

# ============================
# Title
# ============================

st.title("🛒 AI Retail Intelligence")

st.markdown("### AI Powered Sales Prediction Dashboard")

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
    filtered_df.groupby("Sub-Category")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

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



st.markdown("---")

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