import pandas as pd

print("Starting Exploratory Analysis...")

# Load data
df = pd.read_csv(
    r"C:\Users\DELL\OneDrive\Desktop\Projects\E-commerce Customer Behaviour Analysis\cleaned_retail.csv"
)

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.drop(columns=["Unnamed: 8"], errors="ignore")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create Sales column
df["Sales"] = df["Quantity"] * df["UnitPrice"]

# -----------------------------
# Dataset Overview
# -----------------------------
print("\n=== Dataset Overview ===")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Business Metrics
# -----------------------------
total_revenue = df["Sales"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()

print("\n=== Business Summary ===")
print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")

# -----------------------------
# Top Products
# -----------------------------
print("\n=== Top 10 Products by Revenue ===")

top_products = (
    df.groupby("Description")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_products)

# -----------------------------
# Top Countries
# -----------------------------
print("\n=== Top Countries by Revenue ===")

top_countries = (
    df.groupby("Country")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_countries)

# -----------------------------
# Top Customers
# -----------------------------
print("\n=== Top 10 Customers by Revenue ===")

top_customers = (
    df.groupby("CustomerID")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_customers)

# -----------------------------
# Monthly Revenue Trend
# -----------------------------
monthly_revenue = (
    df.groupby(df["InvoiceDate"].dt.to_period("M"))["Sales"]
      .sum()
)

print("\n=== Monthly Revenue Trend ===")
print(monthly_revenue)

# -----------------------------
# Customer Purchase Frequency
# -----------------------------
purchase_frequency = (
    df.groupby("CustomerID")["InvoiceNo"]
      .nunique()
      .sort_values(ascending=False)
)

print("\n=== Most Active Customers ===")
print(purchase_frequency.head(10))

# -----------------------------
# Average Order Value
# -----------------------------
avg_order_value = total_revenue / total_orders

print("\n=== Customer Value Metrics ===")
print(f"Average Order Value: £{avg_order_value:.2f}")

print("\nExploratory Analysis Completed Successfully.")