from pathlib import Path
from data_cleaning import load_and_clean
from customer_segmentation import build_rfm, run_kmeans, profile_clusters
from churn_analysis import build_churn_dataset, train_churn_model

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail.csv"

print("LOADING DATA...")
df = load_and_clean(DATA_PATH)

print("\nBUILDING RFM...")
rfm = build_rfm(df)

print("\nRUNNING SEGMENTATION...")
rfm_clustered, kmeans = run_kmeans(rfm)
print(profile_clusters(rfm_clustered))

print("\nTRAINING CHURN MODEL...")
X, y = build_churn_dataset(rfm)
churn_model = train_churn_model(X, y)

print("\nSAVING MODELS...")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(churn_model, MODEL_DIR / "churn_model.pkl")
joblib.dump(kmeans, MODEL_DIR / "segmentation_model.pkl")

print("MODELS SAVED SUCCESSFULLY")

print("\nPIPELINE COMPLETE")