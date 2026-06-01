"""
Eurostat Bankruptcy & Macro Data Loader
=========================================
Pulls two datasets:
1. STS_RB_A  — Bankruptcy index by NACE sector and country (annual)
2. nama_10_gdp — GDP growth by country (annual)

The bankruptcy index becomes our default label proxy:
  - High bankruptcy index relative to historical mean = stress/default signal
  - We convert this to a binary distress label for the ML model
"""
import eurostat
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Countries matching our BACH dataset
TARGET_COUNTRIES = ['AT', 'BE', 'HR', 'PT', 'SK']

# NACE sector mapping to our 4 buckets
# STS_RB_A uses aggregate NACE codes
NACE_TO_BUCKET = {
    'A':    'agri_food',
    'B-E':  'manufacturing',
    'C':    'manufacturing',
    'F':    'construction',
    'G-I':  'services',
    'G':    'services',
    'H':    'services',
    'I':    'services',
    'J':    'services',
    'M_N':  'services',
}


def load_bankruptcy_index() -> pd.DataFrame:
    """
    Load and process the Eurostat bankruptcy index (STS_RB_A).
    Returns long-format DataFrame with columns:
    country, year, sector_bucket, bankruptcy_index
    """
    print("Fetching STS_RB_A bankruptcy index...")
    df = eurostat.get_data_df('STS_RB_A')

    # Keep only bankruptcy indicator (not registrations)
    df = df[df['indic_bt'] == 'BKRT']

    # Keep non-seasonally adjusted, index base 2015=100
    df = df[(df['s_adj'] == 'NSA') & (df['unit'] == 'I15')]

    # Rename geo column
    df = df.rename(columns={'geo\\TIME_PERIOD': 'country'})

    # Filter to target countries
    df = df[df['country'].isin(TARGET_COUNTRIES)]

    # Map NACE to sector bucket
    df['sector_bucket'] = df['nace_r2'].map(NACE_TO_BUCKET)
    df = df[df['sector_bucket'].notna()]

    # Melt year columns to long format
    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=['country', 'sector_bucket', 'nace_r2'],
        value_vars=year_cols,
        var_name='year',
        value_name='bankruptcy_index'
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long['bankruptcy_index'] = pd.to_numeric(
        df_long['bankruptcy_index'], errors='coerce'
    )

    # Drop missing
    df_long = df_long.dropna(subset=['bankruptcy_index'])

    # Filter to 2005+ to match BACH
    df_long = df_long[df_long['year'] >= 2005]

    print(f"  Shape: {df_long.shape}")
    print(f"  Countries: {sorted(df_long['country'].unique())}")
    print(f"  Years: {df_long['year'].min()} - {df_long['year'].max()}")
    print(f"  Sectors: {df_long['sector_bucket'].value_counts().to_dict()}")

    return df_long


def create_distress_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bankruptcy index into a binary distress label.

    Logic:
      - For each country+sector, compute rolling 5-year mean and std
      - Label = 1 if bankruptcy_index > mean + 0.5*std (above normal stress)
      - Label = 0 otherwise
      - This captures years of elevated insolvency activity per sector
    """
    df = df.sort_values(['country', 'sector_bucket', 'year'])

    df['bkrt_rolling_mean'] = df.groupby(['country', 'sector_bucket'])[
        'bankruptcy_index'
    ].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    df['bkrt_rolling_std'] = df.groupby(['country', 'sector_bucket'])[
        'bankruptcy_index'
    ].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())

    df['distress_label'] = (
        (df['bankruptcy_index'] >
         df['bkrt_rolling_mean'] + 0.5 * df['bkrt_rolling_std'])
        .astype(int)
    )

    label_dist = df['distress_label'].value_counts()
    pct_distress = df['distress_label'].mean() * 100
    print(f"\nDistress label distribution:")
    print(f"  0 (normal):  {label_dist.get(0, 0):,}")
    print(f"  1 (distress): {label_dist.get(1, 0):,}")
    print(f"  Distress rate: {pct_distress:.1f}%")

    return df


def load_gdp_growth() -> pd.DataFrame:
    """
    Load GDP growth rates by country as a macro feature.
    """
    print("\nFetching GDP growth (nama_10_gdp)...")
    df = eurostat.get_data_df('nama_10_gdp')

    df = df.rename(columns={'geo\\TIME_PERIOD': 'country'})
    df = df[
        (df['country'].isin(TARGET_COUNTRIES)) &
        (df['unit'] == 'CLV_PCH_PRE') &   # % change vs previous year
        (df['na_item'] == 'B1GQ')           # GDP
    ]

    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=['country'],
        value_vars=year_cols,
        var_name='year',
        value_name='gdp_growth_pct'
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long['gdp_growth_pct'] = pd.to_numeric(
        df_long['gdp_growth_pct'], errors='coerce'
    )
    df_long = df_long.dropna(subset=['gdp_growth_pct'])
    df_long = df_long[df_long['year'] >= 2005]

    print(f"  GDP data shape: {df_long.shape}")
    return df_long


if __name__ == "__main__":
    # Load bankruptcy index and create distress label
    bkrt_df = load_bankruptcy_index()
    bkrt_df = create_distress_label(bkrt_df)
    bkrt_df.to_csv(PROCESSED_PATH / "eurostat_distress.csv", index=False)
    print(f"\nSaved: data/processed/eurostat_distress.csv")

    # Load GDP growth
    gdp_df = load_gdp_growth()
    gdp_df.to_csv(PROCESSED_PATH / "eurostat_gdp.csv", index=False)
    print(f"Saved: data/processed/eurostat_gdp.csv")

    # Preview
    print("\nDistress label sample:")
    print(bkrt_df[['country', 'year', 'sector_bucket',
                    'bankruptcy_index', 'distress_label']].head(10).to_string())