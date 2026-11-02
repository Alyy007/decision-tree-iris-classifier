import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Pizza Sales Data Analyzer",
    page_icon="🍕",
    layout="wide"
)

st.title("🍕 Pizza Sales Data Analyzer")
st.write(
    "Pizza Sales CSV ka dashboard: dataset details, sales analysis, "
    "filters aur data-quality errors check karein."
)

# -------------------- Load dataset --------------------
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

uploaded_file = st.sidebar.file_uploader(
    "Pizza sales CSV upload karein (optional)",
    type=["csv"]
)

default_path = Path(__file__).with_name("pizza_sales.csv")
try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    elif default_path.exists():
        df = load_data(default_path)
    else:
        st.error("pizza_sales.csv file app ke sath folder mein rakhein ya sidebar se upload karein.")
        st.stop()
except Exception as e:
    st.error(f"CSV load nahi hui: {e}")
    st.stop()

# -------------------- Clean / prepare --------------------
df.columns = df.columns.str.strip()
required_columns = [
    "pizza_id", "order_id", "pizza_name_id", "quantity", "order_date",
    "order_time", "unit_price", "total_price", "pizza_size",
    "pizza_category", "pizza_ingredients", "pizza_name"
]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(f"Dataset mein required columns missing hain: {missing_columns}")
    st.stop()

data = df.copy()
data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
for col in ["pizza_id", "order_id", "quantity", "unit_price", "total_price"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# -------------------- Sidebar filters --------------------
st.sidebar.header("Filters")
categories = sorted(data["pizza_category"].dropna().astype(str).unique())
sizes = sorted(data["pizza_size"].dropna().astype(str).unique())
selected_categories = st.sidebar.multiselect("Pizza Category", categories, default=categories)
selected_sizes = st.sidebar.multiselect("Pizza Size", sizes, default=sizes)

valid_dates = data["order_date"].dropna()
if not valid_dates.empty:
    date_range = st.sidebar.date_input(
        "Order Date Range",
        value=(valid_dates.min().date(), valid_dates.max().date())
    )
else:
    date_range = None

filtered = data[
    data["pizza_category"].astype(str).isin(selected_categories)
    & data["pizza_size"].astype(str).isin(selected_sizes)
].copy()

if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["order_date"].dt.date.between(start_date, end_date)
    ]

# -------------------- KPIs --------------------
st.subheader("Sales Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${filtered['total_price'].sum():,.2f}")
c2.metric("Total Orders", f"{filtered['order_id'].nunique():,.0f}")
c3.metric("Pizza Sold", f"{filtered['quantity'].sum():,.0f}")
c4.metric("Rows", f"{len(filtered):,}")

st.divider()

# -------------------- Charts --------------------
left, right = st.columns(2)
with left:
    st.subheader("Revenue by Category")
    revenue_cat = filtered.groupby("pizza_category")["total_price"].sum().sort_values()
    if not revenue_cat.empty:
        st.bar_chart(revenue_cat)
    else:
        st.info("Selected filters mein data nahi.")

with right:
    st.subheader("Top 10 Pizzas by Revenue")
    top_pizzas = (
        filtered.groupby("pizza_name")["total_price"].sum()
        .sort_values(ascending=False).head(10)
    )
    if not top_pizzas.empty:
        st.bar_chart(top_pizzas)
    else:
        st.info("Selected filters mein data nahi.")

st.subheader("Revenue Trend")
daily_revenue = (
    filtered.dropna(subset=["order_date"])
    .groupby(filtered.dropna(subset=["order_date"])["order_date"].dt.date)["total_price"]
    .sum()
)
if not daily_revenue.empty:
    st.line_chart(daily_revenue)
else:
    st.info("Date data available nahi.")

# -------------------- Data preview --------------------
st.divider()
st.subheader("Dataset Preview")
st.caption(f"Filtered rows: {len(filtered):,} | Original rows: {len(data):,}")
st.dataframe(filtered, use_container_width=True, height=350)

# -------------------- Data quality / errors --------------------
st.divider()
st.subheader("🧪 Data Quality & Error Detection")
st.write("Neeche common data errors / suspicious records check kiye ja rahe hain.")

checks = {
    "Missing values (blank cells)": int(data.isna().sum().sum()),
    "Duplicate complete rows": int(data.duplicated().sum()),
    "Duplicate pizza_id": int(data["pizza_id"].duplicated().sum()),
    "Invalid / missing order_date": int(data["order_date"].isna().sum()),
    "Non-positive quantity (<= 0)": int((data["quantity"] <= 0).sum() + data["quantity"].isna().sum()),
    "Non-positive unit_price (<= 0)": int((data["unit_price"] <= 0).sum() + data["unit_price"].isna().sum()),
    "Non-positive total_price (<= 0)": int((data["total_price"] <= 0).sum() + data["total_price"].isna().sum()),
    "Revenue mismatch (quantity × unit_price)": int(
        (data["total_price"].notna() & data["quantity"].notna() & data["unit_price"].notna()
         & ~np.isclose(data["total_price"], data["quantity"] * data["unit_price"], atol=0.01)).sum()
    ),
}
checks_df = pd.DataFrame({"Check": list(checks.keys()), "Issue Count": list(checks.values())})
st.dataframe(checks_df, use_container_width=True, hide_index=True)

# Build rows that contain at least one detectable issue
issues = pd.Series(False, index=data.index)
issues |= data.isna().any(axis=1)
issues |= data.duplicated(keep=False)
issues |= data["pizza_id"].duplicated(keep=False)
issues |= data["order_date"].isna()
issues |= data["quantity"].isna() | (data["quantity"] <= 0)
issues |= data["unit_price"].isna() | (data["unit_price"] <= 0)
issues |= data["total_price"].isna() | (data["total_price"] <= 0)
valid_money = data["total_price"].notna() & data["quantity"].notna() & data["unit_price"].notna()
issues |= valid_money & ~np.isclose(
    data["total_price"].fillna(0),
    (data["quantity"] * data["unit_price"]).fillna(0),
    atol=0.01
)

error_rows = data.loc[issues].copy()
st.metric("Rows with at least one issue", f"{len(error_rows):,}")
st.dataframe(error_rows, use_container_width=True, height=300)

if not error_rows.empty:
    csv_errors = error_rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download detected error rows (CSV)",
        data=csv_errors,
        file_name="pizza_sales_error_rows.csv",
        mime="text/csv"
    )
else:
    st.success("Is dataset mein in checks ke mutabiq koi issue detect nahi hua.")

# -------------------- Dataset information --------------------
with st.expander("Dataset ki complete details"):
    st.write("**Shape:**", f"{data.shape[0]:,} rows × {data.shape[1]} columns")
    st.write("**Columns:**", list(data.columns))
    st.write("**Data types:**")
    st.dataframe(data.dtypes.astype(str).rename("Data Type").to_frame())
    st.write("**Statistical summary (numeric):**")
    st.dataframe(data.describe().T, use_container_width=True)
    st.write("**Missing values by column:**")
    st.dataframe(data.isna().sum().rename("Missing Count").to_frame())
    st.write("**Category values:**")
    for col in ["pizza_category", "pizza_size"]:
        st.write(f"{col}: {sorted(data[col].dropna().astype(str).unique())}")

st.caption("Python • Streamlit • Pandas • Pizza Sales Dataset")
