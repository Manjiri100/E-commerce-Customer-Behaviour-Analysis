import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail.csv"
SEGMENTS_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"
CHURN_PATH = BASE_DIR / "data" / "processed" / "churn_customers.csv"

print("LOADING DATA...")

# =========================
# LOAD TRANSACTION DATA
# =========================
df = pd.read_csv(DATA_PATH)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

df = df.dropna(subset=["CustomerID"])

df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

df["TotalPrice"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

# =========================
# BUSINESS KPIs
# =========================
total_revenue = df["TotalPrice"].sum()

total_customers = df["CustomerID"].nunique()

total_orders = df["InvoiceNo"].nunique()

avg_order_value = (
    total_revenue /
    total_orders
)

print("\n========== BUSINESS SUMMARY ==========")

print(
    f"\nTotal Revenue: £{total_revenue:,.2f}"
)

print(
    f"Total Customers: {total_customers:,}"
)

print(
    f"Total Orders: {total_orders:,}"
)

print(
    f"Average Order Value: £{avg_order_value:,.2f}"
)

# =========================
# CUSTOMER SEGMENTS
# =========================
if SEGMENTS_PATH.exists():

    segments = pd.read_csv(SEGMENTS_PATH)

    print(
        "\n========== CUSTOMER SEGMENTS =========="
    )

    print(
        segments["Segment"]
        .value_counts()
    )

    print(
        "\nRevenue By Segment:\n"
    )

    print(
        segments.groupby(
            "Segment"
        )["Monetary"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

else:
    print(
        "\ncustomer_segments.csv not found."
    )

# =========================
# CHURN ANALYSIS
# =========================
if CHURN_PATH.exists():

    churn = pd.read_csv(CHURN_PATH)

    customers_at_risk = (
        churn["CustomerID"]
        .nunique()
    )

    revenue_at_risk = (
        churn["Monetary"]
        .sum()
    )

    print(
        "\n========== CHURN ANALYSIS =========="
    )

    print(
        f"\nCustomers At Risk: {customers_at_risk:,}"
    )

    print(
        f"Revenue At Risk: £{revenue_at_risk:,.2f}"
    )

else:
    print(
        "\nchurn_customers.csv not found."
    )

print(
    "\n========== ANALYSIS COMPLETE =========="
)