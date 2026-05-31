"""
Sector-weighted Geopolitical Risk (sGPR) Index
===============================================
Core novel contribution of this project.

Methodology:
    For each Irish NACE sector, compute a weighted average of
    country-level GPR scores using Ireland's bilateral trade
    exposure weights (from CSO Ireland trade statistics).

    sGPR(sector, t) = sum_c [ w(sector, c) * GPR(c, t) ]

    where:
        c  = trading partner country
        w  = sector's share of trade with country c
        t  = time period (monthly)
"""
import pandas as pd
import numpy as np


# Irish trade exposure weights by NACE sector and top trading partners.
# Source: CSO Ireland Trade Statistics — update with actual CSO data.
# These are placeholder weights — replace with real CSO figures in Sprint 3.
SECTOR_TRADE_WEIGHTS = {
    "agri_food": {        # NACE A + C10-C12
        "GB": 0.35, "US": 0.20, "DE": 0.12,
        "FR": 0.10, "NL": 0.08, "other": 0.15
    },
    "manufacturing": {    # NACE C (excl. food)
        "US": 0.30, "DE": 0.18, "GB": 0.15,
        "CN": 0.12, "FR": 0.08, "other": 0.17
    },
    "construction": {     # NACE F
        "GB": 0.40, "DE": 0.15, "PL": 0.12,
        "FR": 0.10, "other": 0.23
    },
    "services": {         # NACE G-N
        "US": 0.35, "GB": 0.25, "DE": 0.12,
        "FR": 0.08, "other": 0.20
    },
}


def compute_sgpr(
    gpr_df: pd.DataFrame,
    sector: str,
    country_col_map: dict | None = None
) -> pd.Series:
    """
    Compute the sector-weighted GPR index for a given sector.

    Parameters
    ----------
    gpr_df : pd.DataFrame
        Monthly GPR data with datetime index and country columns.
        Columns should match keys in SECTOR_TRADE_WEIGHTS values.
    sector : str
        One of: 'agri_food', 'manufacturing', 'construction', 'services'
    country_col_map : dict, optional
        Map from weight keys (e.g. 'GB') to column names in gpr_df.
        If None, assumes column names match weight keys directly.

    Returns
    -------
    pd.Series
        Monthly sGPR index for the sector, same index as gpr_df.
    """
    if sector not in SECTOR_TRADE_WEIGHTS:
        raise ValueError(f"Unknown sector '{sector}'. "
                         f"Choose from: {list(SECTOR_TRADE_WEIGHTS.keys())}")

    weights = SECTOR_TRADE_WEIGHTS[sector]
    sgpr = pd.Series(0.0, index=gpr_df.index, name=f"sgpr_{sector}")
    total_weight = 0.0

    for country, weight in weights.items():
        if country == "other":
            continue
        col = country_col_map.get(country, country) if country_col_map else country
        if col in gpr_df.columns:
            sgpr += weight * gpr_df[col]
            total_weight += weight

    # Renormalise if not all countries were found in gpr_df
    if total_weight > 0 and total_weight < 1.0:
        sgpr = sgpr / total_weight

    return sgpr


def compute_all_sgpr(gpr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sGPR for all sectors and return as a single DataFrame.
    Useful for merging into the master feature matrix.
    """
    results = {}
    for sector in SECTOR_TRADE_WEIGHTS:
        try:
            results[f"sgpr_{sector}"] = compute_sgpr(gpr_df, sector)
        except Exception as e:
            print(f"Warning: could not compute sGPR for {sector}: {e}")
    return pd.DataFrame(results)


def validate_sgpr(sgpr_series: pd.Series, known_stress_dates: list[str]) -> dict:
    """
    Sanity check: verify sGPR spikes around known geopolitical stress events.
    Expected spikes: Brexit vote (2016-06), COVID onset (2020-03),
                     Ukraine invasion (2022-02), 2025 tariff shock.
    """
    results = {}
    for date_str in known_stress_dates:
        date = pd.Timestamp(date_str)
        window = sgpr_series.loc[
            (sgpr_series.index >= date - pd.DateOffset(months=1)) &
            (sgpr_series.index <= date + pd.DateOffset(months=2))
        ]
        if not window.empty:
            results[date_str] = {
                "peak_value": window.max(),
                "peak_date": window.idxmax(),
                "vs_mean": window.max() / sgpr_series.mean()
            }
    return results
