"""
Final Dataset Builder
======================
Merges master dataset with sGPR annual index.
Produces the complete modelling-ready dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def build_final_dataset() -> pd.DataFrame:
    print("Loading datasets...")
    master = pd.read_csv(PROCESSED_PATH / "master_dataset.csv")
    sgpr = pd.read_csv(PROCESSED_PATH / "sgpr_annual.csv")

    print(f"  Master:      {master.shape}")
    print(f"  sGPR annual: {sgpr.shape}")

    # Merge on year only — sGPR is sector-specific so we match
    # each row's sector_bucket to the corresponding sGPR column
    df = master.merge(sgpr, on='year', how='left')
    print(f"\nAfter sGPR merge: {df.shape}")

    # Extract the sector-specific sGPR for each row
    # Instead of keeping all 4 sGPR columns, pick the one matching the row's sector
    def get_sector_sgpr(row):
        col = f"sgpr_{row['sector_bucket']}"
        return row[col] if col in row.index else np.nan

    def get_sector_sgpr_lag(row):
        col = f"sgpr_{row['sector_bucket']}_lag1"
        return row[col] if col in row.index else np.nan

    print("Extracting sector-matched sGPR values...")
    df['sgpr'] = df.apply(get_sector_sgpr, axis=1)
    df['sgpr_lag1'] = df.apply(get_sector_sgpr_lag, axis=1)

    # Drop the individual sGPR sector columns — we only need the matched one
    sgpr_drop = [c for c in df.columns if c.startswith('sgpr_agri') or
                 c.startswith('sgpr_manu') or c.startswith('sgpr_const') or
                 c.startswith('sgpr_serv')]
    df = df.drop(columns=sgpr_drop)

    # Add Brexit dummy (2016-2020 transition period)
    df['brexit_period'] = df['year'].isin([2016, 2017, 2018, 2019, 2020]).astype(int)

    # Add GPR normalised (z-score) for comparability
    df['gpr_zscore'] = (
        (df['gpr_global'] - df['gpr_global'].mean()) /
        df['gpr_global'].std()
    )

    # Drop rows missing sGPR (years outside GPR coverage)
    before = len(df)
    df = df.dropna(subset=['sgpr', 'sgpr_lag1'])
    print(f"After dropping missing sGPR: {len(df):,} rows (dropped {before-len(df)})")

    # Final feature list
    financial_features = [c for c in df.columns
                          if c.startswith('r') and c.endswith('_wm')]
    geo_features = ['sgpr', 'sgpr_lag1', 'gpr_global', 'gpr_zscore']
    macro_features = ['gdp_growth_pct', 'crisis_period',
                      'post_covid', 'brexit_period']

    print(f"\n{'='*55}")
    print(f"FINAL MODELLING DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"Shape:             {df.shape}")
    print(f"Countries:         {sorted(df['country'].unique())}")
    print(f"Years:             {df['year'].min()} - {df['year'].max()}")
    print(f"Sectors:           {df['sector_bucket'].value_counts().to_dict()}")
    print(f"Distress rate:     {df['distress_label'].mean()*100:.1f}%")
    print(f"\nFeature sets:")
    print(f"  Financial ({len(financial_features)}): {financial_features}")
    print(f"  Geopolitical ({len(geo_features)}): {geo_features}")
    print(f"  Macro ({len(macro_features)}): {macro_features}")
    print(f"  Label: distress_label")
    print(f"\nTotal features: {len(financial_features)+len(geo_features)+len(macro_features)}")

    return df, financial_features, geo_features, macro_features


if __name__ == "__main__":
    df, fin_feats, geo_feats, macro_feats = build_final_dataset()

    # Save
    output_path = PROCESSED_PATH / "final_modelling_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Quick correlation check — do geo features correlate with distress?
    print("\nCorrelation of geopolitical features with distress label:")
    all_features = fin_feats + geo_feats + macro_feats
    corr = df[all_features + ['distress_label']].corr()['distress_label']
    corr_sorted = corr.drop('distress_label').abs().sort_values(ascending=False)
    print(corr_sorted.to_string())