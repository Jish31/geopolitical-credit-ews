"""
Master Dataset Builder
=======================
Merges BACH financial ratios + Eurostat distress labels + GDP growth
into a single modelling-ready dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def build_master_dataset() -> pd.DataFrame:
    """
    Merge all processed data sources into one modelling dataset.
    """
    print("Loading processed datasets...")
    bach = pd.read_csv(PROCESSED_PATH / "bach_processed.csv")
    distress = pd.read_csv(PROCESSED_PATH / "eurostat_distress.csv")
    gdp = pd.read_csv(PROCESSED_PATH / "eurostat_gdp.csv")

    print(f"  BACH:     {bach.shape}")
    print(f"  Distress: {distress.shape}")
    print(f"  GDP:      {gdp.shape}")

    # ── Step 1: Merge BACH + distress label ──────────────────
    distress_slim = distress[[
        'country', 'year', 'sector_bucket', 'distress_label', 'bankruptcy_index'
    ]].drop_duplicates()

    df = bach.merge(
        distress_slim,
        on=['country', 'year', 'sector_bucket'],
        how='inner'
    )
    print(f"\nAfter BACH + distress merge: {df.shape}")

    # ── Step 2: Merge GDP growth ──────────────────────────────
    df = df.merge(gdp, on=['country', 'year'], how='left')
    print(f"After GDP merge: {df.shape}")

    # ── Step 3: Drop high-missing columns ────────────────────
    drop_cols = ['r28_wm', 'r29_wm']  # 50.9% missing
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Step 4: Drop rows with missing ratio values ───────────
    ratio_cols = [c for c in df.columns if c.startswith('r') and c.endswith('_wm')]
    before = len(df)
    df = df.dropna(subset=ratio_cols)
    print(f"After dropping missing ratios: {len(df):,} rows (dropped {before-len(df)})")

    # ── Step 5: Winsorise extreme outliers ───────────────────
    # Cap at 1st and 99th percentile per ratio column
    for col in ratio_cols:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p1, upper=p99)
    print(f"Winsorised {len(ratio_cols)} ratio columns at 1st/99th percentile")

    # ── Step 6: Add time features ────────────────────────────
    df['crisis_period'] = df['year'].isin([2008, 2009, 2020]).astype(int)
    df['post_covid'] = (df['year'] >= 2021).astype(int)

    # ── Step 7: Final cleanup ────────────────────────────────
    df = df.sort_values(['country', 'sector_bucket', 'year']).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"MASTER DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Shape:        {df.shape}")
    print(f"Countries:    {sorted(df['country'].unique())}")
    print(f"Years:        {df['year'].min()} - {df['year'].max()}")
    print(f"Sectors:      {df['sector_bucket'].value_counts().to_dict()}")
    print(f"Distress rate: {df['distress_label'].mean()*100:.1f}%")
    print(f"\nFeature columns:")
    print(f"  Financial ratios: {ratio_cols}")
    print(f"  Macro: ['gdp_growth_pct', 'crisis_period', 'post_covid']")
    print(f"  Label: distress_label")

    return df


if __name__ == "__main__":
    df = build_master_dataset()
    output_path = PROCESSED_PATH / "master_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print("\nFirst 5 rows:")
    print(df[['country', 'year', 'sector_bucket',
              'r11_wm', 'r21_wm', 'r31_wm',
              'gdp_growth_pct', 'distress_label']].head().to_string())