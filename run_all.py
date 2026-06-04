from pathlib import Path

from data_cleaning import load_and_clean
from eda import run_eda
from customer_segmentation import build_rfm, run_kmeans
from churn_analysis import run_churn_analysis


# -----------------------------
# Base path (safe for all systems)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail.csv"


def main():
 print("\n🚀 STARTING PIPELINE\n")

df = load_and_clean(DATA_PATH)

print("\n📊 Building RFM...")
rfm = build_rfm(df)

print("\n👥 Running Segmentation...")
rfm_clustered, kmeans_model = run_kmeans(rfm)

print("\n⚠️ Running Churn Analysis...")
run_churn_analysis(rfm_clustered)

print("\n✅ PIPELINE COMPLETE")   


if __name__ == "__main__":
    main()