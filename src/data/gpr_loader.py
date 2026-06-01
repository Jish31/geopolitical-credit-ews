"""
GPR Index Loader & sGPR Constructor
=====================================
Loads Caldara & Iacoviello GPR index and constructs
the sector-weighted Geopolitical Risk (sGPR) index
for Irish NACE sectors.

This is the core novel feature of the project.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

# Irish sector trade exposure weights by trading partner.
# Source: CSO Ireland Trade Statistics (proxy weights — update with real CSO data)
# Keys must match GPR country column suffixes (e.g. GBR -> GPRC_GBR)
SECTOR_TRADE_WEIGHTS = {
    'agri_food': {
        'GBR': 0.35, 'USA': 0.20, 'DEU': 0.12,
        'FRA': 0.10, 'NLD': 0.08, 'CHN': 0.05,
        'BEL': 0.05, 'POL': 0.05
    },
    'manufacturing': {
        'USA': 0.30, 'DEU': 0.18, 'GBR': 0.15,
        'CHN': 0.12, 'FRA': 0.08, 'NLD': 0.07,
        'BEL': 0.05, 'JPN': 0.05
    },
    'construction': {
        'GBR': 0.40, 'DEU': 0.15, 'POL': 0.12,
        'FRA': 0.10, 'USA': 0.08, 'RUS': 0.08,
        'CHN': 0.07
    },
    'services': {
        'USA': 0.35, 'GBR': 0.25, 'DEU': 0.12,
        'FRA': 0.08, 'CHN': 0.08, 'IND': 0.07,
        'JPN': 0.05
    }
}


def load_gpr(start_year: int = 2005) -> pd.DataFrame:
    """
    Load GPR index, parse dates, filter to study period.
    Returns monthly DataFrame with global + country GPR columns.
    """
    print("Loading GPR index...")
    df = pd.read_excel(RAW_PATH / "gpr_index.xls")

    # Fix date column — first 1516 rows are real data, rest are metadata
    df = df[df['month'].notna()].copy()
    df = df[df['var_name'].isna()].copy()  # exclude metadata rows

    # Parse month as datetime
    df['date'] = pd.to_datetime(df['month'])
    df = df[df['date'].dt.year >= start_year].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Keep global GPR + all country-specific columns
    country_cols = [c for c in df.columns if c.startswith('GPRC_')]
    keep_cols = ['date', 'GPR'] + country_cols
    df = df[keep_cols]

    # Convert all to numeric
    for col in ['GPR'] + country_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Monthly GPR data: {len(df)} rows")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Country GPR series: {len(country_cols)}")
    print(f"  Global GPR missing: {df['GPR'].isna().sum()} rows")

    return df


def compute_sgpr(gpr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector-weighted GPR (sGPR) for each Irish sector.

    For each sector:
        sGPR(sector, t) = sum_c [ w(sector,c) * GPRC(c, t) ]

    Returns monthly DataFrame with one sGPR column per sector.
    """
    print("\nComputing sGPR index for each sector...")
    result = gpr_df[['date', 'GPR']].copy()

    for sector, weights in SECTOR_TRADE_WEIGHTS.items():
        sgpr = pd.Series(0.0, index=gpr_df.index)
        total_weight = 0.0

        for country, weight in weights.items():
            col = f'GPRC_{country}'
            if col in gpr_df.columns:
                vals = pd.to_numeric(gpr_df[col], errors='coerce')
                sgpr += weight * vals.fillna(vals.median())
                total_weight += weight
            else:
                print(f"  Warning: {col} not found — skipping")

        # Renormalise to account for any missing countries
        if total_weight > 0:
            sgpr = sgpr / total_weight

        result[f'sgpr_{sector}'] = sgpr
        print(f"  sgpr_{sector}: mean={sgpr.mean():.2f}, "
              f"min={sgpr.min():.2f}, max={sgpr.max():.2f}")

    return result


def aggregate_sgpr_annual(sgpr_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly sGPR to annual — using mean of 12 months.
    Also compute lagged versions (t-1 year) as leading indicators.
    """
    sgpr_monthly['year'] = sgpr_monthly['date'].dt.year
    sgpr_cols = [c for c in sgpr_monthly.columns if c.startswith('sgpr_')]

    annual = sgpr_monthly.groupby('year')[['GPR'] + sgpr_cols].mean().reset_index()
    annual.columns = ['year', 'gpr_global'] + sgpr_cols

    # Add 1-year lagged versions — geopolitical risk precedes credit stress
    for col in sgpr_cols + ['gpr_global']:
        annual[f'{col}_lag1'] = annual[col].shift(1)

    print(f"\nAnnual sGPR shape: {annual.shape}")
    print(f"Years: {annual['year'].min()} - {annual['year'].max()}")
    return annual


def validate_sgpr(sgpr_monthly: pd.DataFrame):
    """
    Sanity check: verify sGPR spikes around known stress events.
    Expected: Ukraine invasion (2022), COVID (2020), Brexit (2016).
    """
    print("\nValidation — checking sGPR spikes at known stress events:")
    sgpr_monthly['year'] = sgpr_monthly['date'].dt.year
    baseline = sgpr_monthly[sgpr_monthly['year'].between(2015, 2019)]

    events = {
        'Brexit vote': '2016-06',
        'COVID onset': '2020-03',
        'Ukraine invasion': '2022-02',
    }

    sgpr_cols = [c for c in sgpr_monthly.columns if c.startswith('sgpr_')]

    for event, date_str in events.items():
        row = sgpr_monthly[
            sgpr_monthly['date'] == pd.Timestamp(date_str)
        ]
        if row.empty:
            row = sgpr_monthly[
                sgpr_monthly['date'].dt.to_period('M') ==
                pd.Period(date_str, 'M')
            ]
        if not row.empty:
            print(f"\n  {event} ({date_str}):")
            for col in sgpr_cols:
                val = row[col].values[0]
                base_mean = baseline[col].mean()
                ratio = val / base_mean if base_mean > 0 else 0
                flag = "⚠ SPIKE" if ratio > 1.3 else ""
                print(f"    {col}: {val:.1f} "
                      f"(vs baseline mean {base_mean:.1f}, "
                      f"ratio {ratio:.2f}) {flag}")


if __name__ == "__main__":
    # Load GPR
    gpr_df = load_gpr(start_year=2005)

    # Compute sGPR
    sgpr_monthly = compute_sgpr(gpr_df)

    # Validate against known events
    validate_sgpr(sgpr_monthly)

    # Aggregate to annual
    sgpr_annual = aggregate_sgpr_annual(sgpr_monthly)

    # Save both
    sgpr_monthly.to_csv(PROCESSED_PATH / "sgpr_monthly.csv", index=False)
    sgpr_annual.to_csv(PROCESSED_PATH / "sgpr_annual.csv", index=False)

    print("\nSaved:")
    print("  data/processed/sgpr_monthly.csv")
    print("  data/processed/sgpr_annual.csv")
    print("\nAnnual sGPR sample:")
    print(sgpr_annual[['year', 'gpr_global', 'sgpr_agri_food',
                        'sgpr_manufacturing', 'sgpr_construction',
                        'sgpr_services']].tail(10).to_string())