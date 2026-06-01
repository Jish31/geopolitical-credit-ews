"""
Eurostat Business Death Rate Loader
=====================================
Replaces bankruptcy index as default label.
Uses business death rates by NACE sector and country.
Dataset: BD_9AC_L_FORM_R2
"""
import eurostat
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# All 12 BACH countries
TARGET_COUNTRIES = ['AT', 'BE', 'DE', 'ES', 'FR', 'HR', 'HU', 'IT',
                    'LU', 'PL', 'PT', 'SK']

NACE_TO_BUCKET = {
    'A':    'agri_food',
    'C':    'manufacturing',
    'F':    'construction',
    'G':    'services',
    'H':    'services',
    'I':    'services',
    'J':    'services',
    'M':    'services',
    'N':    'services',
    'G-I':  'services',
    'G-N':  'services',
    'M_N':  'services',
    'B-E':  'manufacturing',
    'B-F':  'manufacturing',
}


def load_business_deaths() -> pd.DataFrame:
    """
    Load Eurostat business death rates by country and NACE sector.
    Death rate = number of deaths / active enterprises * 100
    This is our default label proxy — high death rate = distress year.
    """
    print("Fetching business demography dataset...")

    # Try multiple dataset codes — coverage varies
    datasets_to_try = ['BD_9AC_L_FORM_R2', 'BD_SIZE', 'BD_9BD_SZ_CL_R2']

    df_raw = None
    for code in datasets_to_try:
        try:
            print(f"  Trying {code}...")
            df_raw = eurostat.get_data_df(code)
            print(f"  Success — shape: {df_raw.shape}")
            print(f"  Columns: {list(df_raw.columns[:10])}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if df_raw is None:
        raise RuntimeError("Could not load any business demography dataset")

    return df_raw


def explore_dataset(df: pd.DataFrame):
    """Print dataset structure to understand available dimensions."""
    print("\nDataset exploration:")
    for col in df.columns[:8]:
        try:
            print(f"  {col}: {df[col].unique()[:10]}")
        except:
            print(f"  {col}: (year column)")


if __name__ == "__main__":
    df_raw = load_business_deaths()
    explore_dataset(df_raw)
    df_raw.to_csv(PROCESSED_PATH / "eurostat_deaths_raw.csv", index=False)
    print("\nSaved raw data to data/processed/eurostat_deaths_raw.csv")
    print("Review the output above to identify the correct filters")