"""
BACH Data Loader & Processor
==============================
Loads and filters the BACH full database CSV.
Selects relevant countries, sectors, size classes,
and financial ratio columns for the PD model.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Countries most comparable to Ireland (small open EU economies)
TARGET_COUNTRIES = ['AT', 'BE', 'DE', 'ES', 'FR', 'HR', 'HU', 'IT', 'LU', 'PL', 'PT', 'SK']

# NACE sectors mapped to our 4 Irish sector buckets
SECTOR_MAP = {
    'agri_food':      ['A', 'A01', 'A02', 'A03', 'C10', 'C11', 'C12'],
    'manufacturing':  ['C', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
                       'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25',
                       'C26', 'C27', 'C28', 'C29', 'C30'],
    'construction':   ['F', 'F41', 'F42', 'F43'],
    'services':       ['G', 'G45', 'G46', 'G47', 'H', 'I', 'J', 'M', 'N']
}

# Size classes: 1=small, 1a=micro, 1b=small, 2=medium — all SMEs
SME_SIZES = ['1', '1a', '1b', '2']

# Key financial ratio columns we need for the model
# Prefix meanings: r11=ROA, r12=ROE, r14=leverage, r21=current ratio,
#                  r22=quick ratio, r25=interest coverage, r28=debt ratio
RATIO_COLS = [
    'r11_wm',   # ROA (return on assets) — weighted mean
    'r12_wm',   # ROE (return on equity)
    'r14_wm',   # Gross operating surplus / turnover (profitability)
    'r21_wm',   # Current ratio (liquidity)
    'r22_wm',   # Quick ratio
    'r25_wm',   # Financial debt / EBITDA (leverage)
    'r28_wm',   # Debt ratio (total debt / total assets)
    'r29_wm',   # Financial debt ratio
    'r31_wm',   # Interest coverage
    'r32_wm',   # Net debt / equity
    'r51_wm',   # Asset turnover
    'r52_wm',   # Receivables days
    'r53_wm',   # Payables days
]

ID_COLS = ['country', 'year', 'sector', 'size', 'sample',
           'total_assets', 'turnover', 'gross_value_added', 'nb_firms']


def load_bach_filtered() -> pd.DataFrame:
    """
    Load and filter BACH data to SME-relevant subset.
    Returns a clean DataFrame ready for feature engineering.
    """
    print("Loading BACH full dataset...")
    df = pd.read_csv(
        RAW_PATH / "bach_full.csv",
        sep=';', skiprows=1, low_memory=False
    )
    print(f"  Full shape: {df.shape}")

    # Filter to target countries
    df = df[df['country'].isin(TARGET_COUNTRIES)]
    print(f"  After country filter ({TARGET_COUNTRIES}): {len(df):,} rows")

    # Filter to SME size classes only
    df = df[df['size'].isin(SME_SIZES)]
    print(f"  After SME size filter: {len(df):,} rows")

    # Filter to sample == 1 (representative sample only, exclude -1/0)
    df = df[df['sample'] == 1]
    print(f"  After sample quality filter: {len(df):,} rows")

    # Filter to years 2005+ (post-harmonisation, more reliable)
    df = df[df['year'] >= 2005]
    print(f"  After year filter (>=2005): {len(df):,} rows")

    # Add sector bucket label
    sector_to_bucket = {}
    for bucket, sectors in SECTOR_MAP.items():
        for s in sectors:
            sector_to_bucket[s] = bucket
    df['sector_bucket'] = df['sector'].map(sector_to_bucket)

    # Keep only rows with a recognised sector bucket
    df = df[df['sector_bucket'].notna()]
    print(f"  After sector filter: {len(df):,} rows")

    # Select only the columns we need
    available_ratios = [c for c in RATIO_COLS if c in df.columns]
    keep_cols = ID_COLS + ['sector_bucket'] + available_ratios
    df = df[[c for c in keep_cols if c in df.columns]]

    # Convert ratio columns to numeric
    for col in available_ratios:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\nFinal shape: {df.shape}")
    print(f"Countries: {sorted(df['country'].unique())}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Sector buckets: {df['sector_bucket'].value_counts().to_dict()}")
    print(f"\nRatio columns available: {available_ratios}")

    return df


def save_processed(df: pd.DataFrame, filename: str = "bach_processed.csv"):
    path = PROCESSED_PATH / filename
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    df = load_bach_filtered()

    print("\nSample of ratio data (first 3 rows):")
    ratio_cols = [c for c in RATIO_COLS if c in df.columns]
    print(df[['country', 'year', 'sector_bucket'] + ratio_cols[:5]].head(3).to_string())

    print("\nMissing values in ratio columns:")
    for col in ratio_cols:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"  {col}: {pct:.1f}% missing")

    save_processed(df)