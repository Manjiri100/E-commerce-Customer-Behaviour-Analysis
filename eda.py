import pandas as pd
import matplotlib.pyplot as plt

from utils import DATA_PATH, FIG_DIR
from data_cleaning import load_and_clean


# -----------------------------
# EDA PIPELINE FUNCTION
# -----------------------------
def run_eda(df=None):
    print("\n📊 STARTING EDA...\n")

    # If df not passed, load it
    if df is None:
        df = load_and_clean(DATA_PATH)

    # Ensure figures folder exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- KPIs ----------------
    print("\n========== KPIs ==========")

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    kpis = {
        "total_revenue": df["TotalPrice"].sum(),
        "total_customers": df["CustomerID"].nunique(),
        "total_orders": df["InvoiceNo"].nunique()
    }

    print("TOTAL REVENUE:", round(kpis["total_revenue"], 2))
    print("TOTAL CUSTOMERS:", kpis["total_customers"])
    print("TOTAL ORDERS:", kpis["total_orders"])

    # ---------------- TOP CUSTOMERS ----------------
    print("\n========== TOP CUSTOMERS ==========")

    top_cust = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    print(top_cust)

    plt.figure()
    top_cust.plot(kind="bar")
    plt.title("Top Customers")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_customers.png")
    plt.close()

    # ---------------- MONTHLY SALES ----------------
    print("\n========== MONTHLY SALES ==========")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    monthly = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()

    print(monthly.head())

    plt.figure()
    monthly.plot()
    plt.title("Monthly Sales Trend")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_sales.png")
    plt.close()

    # ---------------- COUNTRY SALES ----------------
    print("\n========== COUNTRY SALES ==========")

    country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    print(country.head(10))

    print("\n✅ EDA COMPLETED")


# -----------------------------
# Standalone run support
# -----------------------------
if __name__ == "__main__":
    run_eda()