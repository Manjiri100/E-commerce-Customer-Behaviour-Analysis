# src/utils.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail.csv"

REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR = REPORTS_DIR / "outputs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
