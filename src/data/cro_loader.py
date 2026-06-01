"""
CRO Ireland Financial Statements Loader
========================================
Source: https://opendata.cro.ie
Licence: CC BY 4.0
"""
import requests
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

RESOURCES = {
    "2022": {
        "id": "508d4f8a-74a1-40c7-8b86-cdf0d54a4929",
        "url": "https://opendata.cro.ie/dataset/99e64d94-0a2f-4cd1-b237-7164bec1426e/resource/508d4f8a-74a1-40c7-8b86-cdf0d54a4929/download/financial_statements.csv"
    },
    "2023": {
        "id": "dd413039-f628-4931-9788-dfc38eaf6b99",
        "url": "https://opendata.cro.ie/dataset/99e64d94-0a2f-4cd1-b237-7164bec1426e/resource/dd413039-f628-4931-9788-dfc38eaf6b99/download/financial_statements_2023.csv"
    }
}


def download_csv(year: str) -> pd.DataFrame:
    """Download full financial statements CSV for a given year."""
    save_path = RAW_PATH / f"cro_financial_statements_{year}.csv"

    if save_path.exists():
        print(f"  [{year}] Already downloaded — loading from disk.")
        return pd.read_csv(save_path, low_memory=False)

    print(f"  [{year}] Downloading from CRO portal...")
    url = RESOURCES[year]["url"]
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"  [{year}] Saved to {save_path}")

    df = pd.read_csv(save_path, low_memory=False)
    return df


def explore_columns(df: pd.DataFrame, year: str):
    """Print all columns and a sample row to understand the data structure."""
    print(f"\n{'='*60}")
    print(f"  Year: {year} | Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"{'='*60}")
    print("\nAll columns:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nSample row:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    for year in ["2022", "2023"]:
        df = download_csv(year)
        explore_columns(df, year)