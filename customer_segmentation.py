import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("cleaned_retail.csv")

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# RFM Calculation
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": "max",
    "InvoiceNo": "count",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# Scale data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Frequency", "Monetary"]])

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Segment"] = kmeans.fit_predict(rfm_scaled)

print("\n👥 CUSTOMER SEGMENTS (Sample)")
print(rfm.head())

print("✅ Customer segmentation completed.")
