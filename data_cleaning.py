import pandas as pd

def load_and_clean(path):
    print("LOOKING FOR FILE:", path)

    df = pd.read_csv(path)

    print("FILE EXISTS:", True)
    print("SUCCESS: Data loaded")

    # Remove junk columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Clean missing values
    df = df.dropna(subset=["CustomerID"])

    # Remove invalid transactions
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Fix types
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Feature engineering
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df