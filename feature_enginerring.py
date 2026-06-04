import pandas as pd

def create_features(df):

    snapshot_date = df["InvoiceDate"].max()

    customer_df = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum",
        "Quantity": "sum"
    })

    customer_df.columns = ["Recency", "Frequency", "Monetary", "TotalQuantity"]

    # Add behavioral features
    customer_df["AvgOrderValue"] = customer_df["Monetary"] / customer_df["Frequency"]

    return customer_df.reset_index()