"""
cleaning.py
===========
Data cleaning and preprocessing pipeline for petroleum production data.

Functions handle:
  - Missing value imputation (well-specific forward-fill + median fallback)
  - Outlier detection and capping using IQR method
  - Data type enforcement
  - Feature engineering (month, quarter, year columns)
  - Validation checks and data quality report

Usage (standalone):
    python src/cleaning.py

Usage (as module):
    from src.cleaning import load_and_clean
    df_clean = load_and_clean("data/raw_production_data.csv")
"""

import pandas as pd
import numpy as np
import os


# ── Constants ──────────────────────────────────────────────────────────────────
NUMERIC_COLS = ["oil_rate_bbl_day", "water_cut_pct", "wellhead_pressure_psi"]
RATE_FLOOR   = 1.0    # bbl/day — physical minimum
RATE_CEIL    = 5000.0 # bbl/day — physical maximum (ultra-high rate cap)


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and enforce correct data types.

    Args:
        filepath: Path to raw_production_data.csv

    Returns:
        pd.DataFrame with date parsed and categoricals set
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df["well_id"]      = df["well_id"].astype("category")
    df["basin"]        = df["basin"].astype("category")
    df["field"]        = df["field"].astype("category")
    df["decline_type"] = df["decline_type"].astype("category")
    df["anomaly_flag"] = df["anomaly_flag"].astype(int)
    print(f"[load]    Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def report_missing(df: pd.DataFrame) -> None:
    """Print a data quality report showing missing values per column."""
    missing = df[NUMERIC_COLS].isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    print("\n[quality] Missing Value Report:")
    for col in NUMERIC_COLS:
        print(f"          {col:<30} {missing[col]:>4} missing  ({pct[col]}%)")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using a two-step strategy:
      1. Forward-fill within each well (uses last known value — physically sensible)
      2. Backward-fill for any remaining gaps at the start of a well's history
      3. Median fill for any residual nulls (rare)

    Args:
        df: Raw DataFrame with possible NaN in numeric cols

    Returns:
        DataFrame with no missing values in numeric cols
    """
    df = df.sort_values(["well_id", "date"]).copy()

    for col in NUMERIC_COLS:
        # Step 1 & 2: within-well forward then backward fill
        df[col] = df.groupby("well_id", observed=True)[col].transform(
            lambda s: s.ffill().bfill()
        )
        # Step 3: global median fallback (edge case)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[impute]  Missing values after imputation: {df[NUMERIC_COLS].isnull().sum().sum()}")
    return df


def cap_outliers_iqr(df: pd.DataFrame, factor: float = 2.5) -> pd.DataFrame:
    """
    Cap outliers using well-level IQR method.

    For each well and each numeric column:
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
    Values outside this range are clipped to the bounds.

    Note: Anomaly-flagged rows are EXCLUDED from capping — those are
    intentional events we want to preserve for detection.

    Args:
        df     : Cleaned DataFrame
        factor : IQR multiplier (default 2.5 — less aggressive than 1.5)

    Returns:
        DataFrame with outliers capped
    """
    df = df.copy()
    non_anom = df["anomaly_flag"] == 0

    for col in ["oil_rate_bbl_day", "wellhead_pressure_psi"]:
        for well_id_val in df["well_id"].cat.categories:
            mask_well = df["well_id"] == well_id_val
            well_vals = df.loc[mask_well, col]
            q1, q3 = well_vals.quantile(0.25), well_vals.quantile(0.75)
            iqr = q3 - q1
            lo = max(q1 - factor * iqr, RATE_FLOOR)
            hi = min(q3 + factor * iqr, RATE_CEIL)
            mask_normal = mask_well & (df["anomaly_flag"] == 0)
            df.loc[mask_normal, col] = df.loc[mask_normal, col].clip(lo, hi)

    print(f"[outlier] IQR capping applied (factor={factor}) — anomaly rows preserved")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and derived feature columns.

    New columns:
        year              : Calendar year
        month             : Calendar month (1–12)
        quarter           : Q1–Q4
        production_month  : Months since well start (0-based)
        oil_rate_bbl_month: Monthly oil volume (rate × 30 days)
        wc_category       : Water cut bucket (Low / Medium / High / Very High)

    Args:
        df: Imputed and capped DataFrame

    Returns:
        DataFrame with new feature columns
    """
    df = df.copy()
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Months since each well's first production record
    well_start = df.groupby("well_id", observed=True)["date"].transform("min")
    df["production_month"] = (
        (df["date"].dt.year  - well_start.dt.year) * 12 +
        (df["date"].dt.month - well_start.dt.month)
    )

    # Monthly volume proxy
    df["oil_rate_bbl_month"] = (df["oil_rate_bbl_day"] * 30).round(0)

    # Water cut category
    bins   = [0, 25, 50, 75, 100]
    labels = ["Low (<25%)", "Medium (25-50%)", "High (50-75%)", "Very High (>75%)"]
    df["wc_category"] = pd.cut(df["water_cut_pct"], bins=bins, labels=labels)

    print(f"[feature] Feature engineering complete — {df.shape[1]} total columns")
    return df


def validate(df: pd.DataFrame) -> bool:
    """
    Run data validation checks and report results.

    Checks:
        - No missing values in numeric cols
        - oil_rate > 0 for all rows
        - water_cut in [0, 100]
        - production_month is non-negative

    Returns:
        True if all checks pass, False otherwise
    """
    checks = {
        "No NaN in numeric cols"     : df[NUMERIC_COLS].isnull().sum().sum() == 0,
        "oil_rate_bbl_day > 0"       : (df["oil_rate_bbl_day"] > 0).all(),
        "water_cut in [0, 100]"      : df["water_cut_pct"].between(0, 100).all(),
        "production_month >= 0"      : (df["production_month"] >= 0).all(),
    }
    print("\n[validate] Validation Results:")
    all_pass = True
    for check, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"           {status}  {check}")
        if not result:
            all_pass = False
    return all_pass


def load_and_clean(filepath: str, output_path: str = "data/cleaned_production_data.csv") -> pd.DataFrame:
    """
    Master pipeline: load → validate missing → impute → cap → engineer → validate.

    Args:
        filepath    : Path to raw CSV
        output_path : Where to save cleaned CSV

    Returns:
        Fully cleaned and feature-engineered DataFrame
    """
    print("=" * 55)
    print(" PETROLEUM DATA CLEANING PIPELINE")
    print("=" * 55)

    df = load_raw(filepath)
    report_missing(df)
    df = impute_missing(df)
    df = cap_outliers_iqr(df)
    df = engineer_features(df)
    validate(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[save]    Cleaned data saved → {output_path}")
    print(f"          Final shape: {df.shape}")
    print("=" * 55)
    return df


if __name__ == "__main__":
    df_clean = load_and_clean("data/raw_production_data.csv")
    print("\nSample cleaned data:")
    print(df_clean[["date", "well_id", "basin", "oil_rate_bbl_day",
                     "water_cut_pct", "production_month", "wc_category"]].head(8).to_string())
