import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("cleaned_retail.csv", parse_dates=["InvoiceDate"])

df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]

df["CustomerID"] = df["CustomerID"].astype(int)

# Revenue feature
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# -----------------------
# 2. RFM Feature Engineering
# -----------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

# -----------------------
# 3. Scaling
# -----------------------
features = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

# -----------------------
# 4. Find Best K
# -----------------------
silhouette_scores = {}

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    silhouette_scores[k] = silhouette_score(rfm_scaled, labels)

best_k = max(silhouette_scores, key=silhouette_scores.get)

print("\nSilhouette Scores:")
for k, v in silhouette_scores.items():
    print(f"k={k}: {v:.4f}")

print(f"\nBest number of clusters: {best_k}")

# -----------------------
# 5. Final Model
# -----------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["Segment"] = kmeans.fit_predict(rfm_scaled)

print("\nSample Output:")
print(rfm.head())

# -----------------------
# 6. Segment Summary
# -----------------------
segment_summary = rfm.groupby("Segment")[features].mean().reset_index()

print("\nSEGMENT PROFILE")
print(segment_summary)

# -----------------------
# 7. Business Insight
# -----------------------
for _, row in segment_summary.iterrows():
    print(
        f"\nSegment {int(row['Segment'])}: "
        f"Recency={row['Recency']:.1f}, "
        f"Frequency={row['Frequency']:.1f}, "
        f"Monetary={row['Monetary']:.1f}"
    )

print("\nRecommendation:")
print("Focus retention on high-value low-recency customers and upsell high-monetary segments.")
