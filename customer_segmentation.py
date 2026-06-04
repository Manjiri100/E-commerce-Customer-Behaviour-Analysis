import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def build_rfm(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

    snapshot = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm


def run_kmeans(rfm, n_clusters=4):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    return rfm, kmeans


def profile_clusters(rfm):
    return rfm.groupby("Cluster").mean().sort_values("Monetary", ascending=False)
def run_segmentation(df):
    print("\n👥 STARTING CUSTOMER SEGMENTATION...\n")

    # Step 1: Build RFM
    rfm = build_rfm(df)

    # Step 2: Run KMeans
    rfm_clustered, model = run_kmeans(rfm)

    # Step 3: Profile clusters
    summary = profile_clusters(rfm_clustered)

    print("\n📊 CLUSTER SUMMARY:")
    print(summary)

    print("\n✅ SEGMENTATION COMPLETE")

    return rfm_clustered, model, summary