"""
Full Dataset Builder v2
========================
Uses business death rate (V97020) as distress label.
Covers all 12 BACH countries + Ireland, 2008-2020.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

TARGET_COUNTRIES = ['AT','BE','DE','ES','FR','HR','HU','IT','LU','PL','PT','SK','IE']

NACE_TO_BUCKET = {
    'A':   'agri_food',
    'C':   'manufacturing',
    'B-E': 'manufacturing',
    'F':   'construction',
    'G':   'services',
    'H':   'services',
    'I':   'services',
    'J':   'services',
    'M':   'services',
    'N':   'services',
    'G-N_X_K642': 'services',
}

FIN = ['r11_wm','r12_wm','r14_wm','r21_wm','r22_wm',
       'r25_wm','r31_wm','r32_wm','r51_wm','r52_wm','r53_wm']


# ── Step 1: Load BACH (all 12 countries) ─────────────────
def load_bach():
    print("Loading BACH...")
    df = pd.read_csv('data/raw/bach_full.csv', sep=';', skiprows=1, low_memory=False)

    SECTOR_MAP = {
        'agri_food':    ['A','A01','A02','A03','C10','C11','C12'],
        'manufacturing':['C','C13','C14','C15','C16','C17','C18',
                         'C19','C20','C21','C22','C23','C24','C25',
                         'C26','C27','C28','C29','C30'],
        'construction': ['F','F41','F42','F43'],
        'services':     ['G','G45','G46','G47','H','I','J','M','N']
    }
    sector_to_bucket = {s: b for b, ss in SECTOR_MAP.items() for s in ss}

    df = df[df['size'].isin(['1','1a','1b','2'])]
    df = df[df['sample'] == 1]
    df = df[df['year'] >= 2008]
    df['sector_bucket'] = df['sector'].map(sector_to_bucket)
    df = df[df['sector_bucket'].notna()]

    available_fin = [c for c in FIN if c in df.columns]
    keep = ['country','year','sector','size','sector_bucket',
            'total_assets','turnover','nb_firms'] + available_fin
    df = df[[c for c in keep if c in df.columns]]

    for col in available_fin:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Aggregate to country-sector-year
    agg = df.groupby(['country','sector_bucket','year']).agg(
        **{f: (f,'mean') for f in available_fin},
        nb_firms=('nb_firms','sum')
    ).reset_index()

    print(f"  BACH aggregated: {agg.shape}")
    return agg


# ── Step 2: Load death rate label ────────────────────────
def load_death_rate():
    print("Loading business death rate...")
    df = pd.read_csv('data/processed/eurostat_deaths_raw.csv')
    df = df.rename(columns={'geo\\TIME_PERIOD': 'country'})

    # Keep death rate indicator, all enterprises, target countries
    df = df[
        (df['indic_sb'] == 'V97020') &
        (df['leg_form'] == 'TOTAL') &
        (df['country'].isin(TARGET_COUNTRIES))
    ]

    # Map NACE to sector bucket
    df['sector_bucket'] = df['nace_r2'].map(NACE_TO_BUCKET)
    df = df[df['sector_bucket'].notna()]

    # Melt to long format
    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(
        id_vars=['country','sector_bucket','nace_r2'],
        value_vars=year_cols,
        var_name='year',
        value_name='death_rate'
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long['death_rate'] = pd.to_numeric(df_long['death_rate'], errors='coerce')
    df_long = df_long.dropna(subset=['death_rate'])
    df_long = df_long[df_long['year'] >= 2008]

    # Use top-level NACE only (avoid duplicates from sub-sectors)
    top_nace = ['A','C','F','G','H','I','J','M','N','B-E','G-N_X_K642']
    df_long = df_long[df_long['nace_r2'].isin(top_nace)]

    # Take mean death rate per country-sector-year
    df_agg = df_long.groupby(['country','sector_bucket','year'])['death_rate'].mean().reset_index()

    print(f"  Death rate data: {df_agg.shape}")
    print(f"  Countries: {sorted(df_agg['country'].unique())}")
    print(f"  Years: {df_agg['year'].min()} - {df_agg['year'].max()}")
    return df_agg


# ── Step 3: Create binary distress label ─────────────────
def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 if death_rate > rolling 3yr mean + 0.5*std
    Captures years of abnormally high business deaths per sector/country
    """
    df = df.sort_values(['country','sector_bucket','year'])
    df['dr_roll_mean'] = df.groupby(['country','sector_bucket'])['death_rate'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    df['dr_roll_std'] = df.groupby(['country','sector_bucket'])['death_rate'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    )
    df['distress_label'] = (
        df['death_rate'] > df['dr_roll_mean'] + 0.5 * df['dr_roll_std']
    ).astype(int)
    df = df.dropna(subset=['dr_roll_mean'])
    return df


# ── Step 4: Load GPR and merge ───────────────────────────
def load_sgpr():
    print("Loading sGPR...")
    sgpr = pd.read_csv('data/processed/sgpr_annual.csv')
    print(f"  sGPR: {sgpr.shape}")
    return sgpr


def load_gdp():
    return pd.read_csv('data/processed/eurostat_gdp.csv')


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    bach = load_bach()
    deaths = load_death_rate()
    deaths = create_label(deaths)

    print(f"\nDistress label distribution:")
    print(f"  0: {(deaths['distress_label']==0).sum()}")
    print(f"  1: {(deaths['distress_label']==1).sum()}")
    print(f"  Rate: {deaths['distress_label'].mean()*100:.1f}%")

    # Merge BACH + death rate label
    df = bach.merge(
        deaths[['country','sector_bucket','year',
                'death_rate','distress_label']],
        on=['country','sector_bucket','year'],
        how='inner'
    )
    print(f"\nAfter BACH + label merge: {df.shape}")

    # Merge sGPR
    sgpr = load_sgpr()
    df = df.merge(sgpr, on='year', how='left')

    # Extract sector-matched sGPR
    def get_sgpr(row):
        col = f"sgpr_{row['sector_bucket']}"
        return row.get(col, np.nan)
    def get_sgpr_lag(row):
        col = f"sgpr_{row['sector_bucket']}_lag1"
        return row.get(col, np.nan)

    df['sgpr'] = df.apply(get_sgpr, axis=1)
    df['sgpr_lag1'] = df.apply(get_sgpr_lag, axis=1)

    # Drop individual sgpr sector columns
    drop_cols = [c for c in df.columns if c.startswith('sgpr_agri') or
                 c.startswith('sgpr_manu') or c.startswith('sgpr_const') or
                 c.startswith('sgpr_serv')]
    df = df.drop(columns=drop_cols)

    # Add time features
    df['crisis_2008'] = df['year'].isin([2008,2009]).astype(int)
    df['brexit_period'] = df['year'].isin([2016,2017,2018,2019]).astype(int)
    df['post_covid'] = (df['year'] >= 2020).astype(int)

    # Drop missing
    fin_cols = [c for c in FIN if c in df.columns]
    df = df.dropna(subset=fin_cols + ['sgpr','distress_label'])

    # Winsorise
    for col in fin_cols:
        p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(p1, p99)

    print(f"\n{'='*55}")
    print(f"FINAL DATASET v2 SUMMARY")
    print(f"{'='*55}")
    print(f"Shape:          {df.shape}")
    print(f"Countries:      {sorted(df['country'].unique())}")
    print(f"Years:          {df['year'].min()} - {df['year'].max()}")
    print(f"Sectors:        {df['sector_bucket'].value_counts().to_dict()}")
    print(f"Distress rate:  {df['distress_label'].mean()*100:.1f}%")
    print(f"\nYear x distress:")
    print(pd.crosstab(df['year'], df['distress_label']))

    df.to_csv(PROCESSED_PATH / "final_dataset_v2.csv", index=False)
    print(f"\nSaved: data/processed/final_dataset_v2.csv")