"""
data_generator.py
=================
Generates a synthetic petroleum well production dataset simulating
EIA (U.S. Energy Information Administration) format data.

The dataset mimics real ONGC-style field data across 3 Indian basins:
  - Assam Basin (North-East)
  - Gujarat Basin (West)
  - Andhra / KG Basin (East Coast)

Each well follows Arps decline curve behavior with added noise,
seasonal effects, and injected anomalies (simulating equipment failure
or reservoir damage events).

Usage:
    python src/data_generator.py

Output:
    data/raw_production_data.csv
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Well Configuration ─────────────────────────────────────────────────────────
WELLS = [
    # (well_id, basin,    field,             qi,   Di,   b,    type)
    # qi = initial rate (bbl/day), Di = decline rate (/month), b = hyperbolic exponent
    ("W-01", "Assam",   "Lakwa",           620,  0.10, 0.5, "hyperbolic"),
    ("W-02", "Assam",   "Lakwa",           540,  0.09, 0.4, "hyperbolic"),
    ("W-03", "Assam",   "Geleki",          480,  0.13, 0.0, "exponential"),
    ("W-04", "Assam",   "Geleki",          850,  0.12, 0.0, "exponential"),
    ("W-05", "Gujarat", "Ankleshwar",      710,  0.08, 0.6, "hyperbolic"),
    ("W-06", "Gujarat", "Ankleshwar",      390,  0.11, 0.0, "exponential"),
    ("W-07", "Gujarat", "Gandhar",         560,  0.09, 0.3, "hyperbolic"),
    ("W-08", "Gujarat", "Gandhar",         430,  0.14, 0.0, "exponential"),
    ("W-09", "Andhra",  "KG_Offshore",     320,  0.07, 0.7, "hyperbolic"),
    ("W-10", "Andhra",  "KG_Offshore",     290,  0.10, 0.0, "exponential"),
    ("W-11", "Andhra",  "Ravva",           500,  0.11, 0.5, "hyperbolic"),
    ("W-12", "Andhra",  "Ravva",           410,  0.08, 0.4, "hyperbolic"),
]

MONTHS = 24          # Simulate 2 years of production history
START_DATE = "2023-01-01"

# ── Anomaly Injection Config ───────────────────────────────────────────────────
# (well_id, month_index, factor) — factor < 1 = drop, factor > 1 = spike
ANOMALIES = {
    "W-07": (5,  0.22),   # Severe drop: possible BOP failure / workover needed
    "W-03": (8,  0.18),   # Severe drop: reservoir damage
    "W-11": (3,  1.65),   # Spike: possibly cross-flow from adjacent zone
}


def arps_exponential(qi: float, Di: float, t: np.ndarray) -> np.ndarray:
    """
    Arps Exponential Decline Curve.

    Formula:  q(t) = qi * exp(-Di * t)

    Args:
        qi  : Initial production rate (bbl/day)
        Di  : Nominal decline rate (per month)
        t   : Array of time steps (months)

    Returns:
        Production rate array (bbl/day)
    """
    return qi * np.exp(-Di * t)


def arps_hyperbolic(qi: float, Di: float, b: float, t: np.ndarray) -> np.ndarray:
    """
    Arps Hyperbolic Decline Curve.

    Formula:  q(t) = qi / (1 + b * Di * t)^(1/b)

    Args:
        qi  : Initial production rate (bbl/day)
        Di  : Nominal decline rate (per month)
        b   : Hyperbolic exponent (0 < b < 1)
        t   : Array of time steps (months)

    Returns:
        Production rate array (bbl/day)
    """
    return qi / np.power(1 + b * Di * t, 1 / b)


def seasonal_factor(month_index: int) -> float:
    """
    Applies a seasonal multiplier to simulate:
    - Monsoon slowdown (Jun–Sep, months 5–8): ~8% reduction
    - Winter peak (Nov–Jan): ~5% boost
    - Normal otherwise

    Args:
        month_index: 0-based month index

    Returns:
        Multiplicative factor (float near 1.0)
    """
    month_of_year = month_index % 12
    if 5 <= month_of_year <= 8:     # Monsoon
        return np.random.uniform(0.89, 0.95)
    elif month_of_year in [10, 11, 0]:  # Winter
        return np.random.uniform(1.03, 1.07)
    else:
        return np.random.uniform(0.97, 1.03)


def generate_water_cut(t: np.ndarray, basin: str) -> np.ndarray:
    """
    Simulates water cut (%) — increases over time as reservoir depletes.
    Different basins have different initial water cut profiles.

    Args:
        t     : Time array (months)
        basin : Basin name (affects initial water cut)

    Returns:
        Water cut percentage array
    """
    base = {"Assam": 18, "Gujarat": 25, "Andhra": 12}.get(basin, 20)
    wc = base + (t * np.random.uniform(1.2, 2.0))
    noise = np.random.normal(0, 2, len(t))
    return np.clip(wc + noise, 0, 95)


def generate_wellhead_pressure(t: np.ndarray, qi: float) -> np.ndarray:
    """
    Simulates wellhead pressure (psi) — declines with production.

    Args:
        t  : Time array
        qi : Initial rate (used to scale initial pressure)

    Returns:
        Wellhead pressure array (psi)
    """
    initial_p = qi * 0.35 + np.random.uniform(800, 1200)
    pressure = initial_p * np.exp(-0.04 * t)
    noise = np.random.normal(0, 15, len(t))
    return np.clip(pressure + noise, 200, 3500)


def generate_dataset() -> pd.DataFrame:
    """
    Main function: generates the complete multi-well production dataset.

    Returns:
        pd.DataFrame with columns:
            date, well_id, basin, field, oil_rate_bbl_day,
            water_cut_pct, wellhead_pressure_psi, cumulative_oil_bbl,
            anomaly_flag
    """
    records = []
    date_range = pd.date_range(START_DATE, periods=MONTHS, freq="MS")  # Month Start

    for (well_id, basin, field, qi, Di, b, dtype) in WELLS:
        t = np.arange(MONTHS, dtype=float)
        anomaly_flag = [0] * MONTHS

        # ── Base Decline Curve ────────────────────────────────────────────────
        if dtype == "exponential":
            base_rate = arps_exponential(qi, Di, t)
        else:
            base_rate = arps_hyperbolic(qi, Di, b, t)

        # ── Add Noise + Seasonality ───────────────────────────────────────────
        rates = []
        for i, r in enumerate(base_rate):
            season = seasonal_factor(i)
            noise = np.random.normal(0, r * 0.04)   # ±4% operational noise
            rates.append(max(r * season + noise, 5))  # floor at 5 bbl/day

        rates = np.array(rates)

        # ── Inject Anomalies ──────────────────────────────────────────────────
        if well_id in ANOMALIES:
            anom_month, factor = ANOMALIES[well_id]
            rates[anom_month] = rates[anom_month] * factor
            anomaly_flag[anom_month] = 1

        # ── Supporting Parameters ─────────────────────────────────────────────
        water_cut = generate_water_cut(t, basin)
        pressure  = generate_wellhead_pressure(t, qi)

        # ── Cumulative Production (trapezoidal integration × 30 days/month) ──
        cumulative = np.cumsum(rates * 30)

        # ── Assemble Records ──────────────────────────────────────────────────
        for i in range(MONTHS):
            records.append({
                "date":                   date_range[i],
                "well_id":                well_id,
                "basin":                  basin,
                "field":                  field,
                "decline_type":           dtype,
                "oil_rate_bbl_day":       round(rates[i], 2),
                "water_cut_pct":          round(water_cut[i], 2),
                "wellhead_pressure_psi":  round(pressure[i], 1),
                "cumulative_oil_bbl":     round(cumulative[i], 0),
                "anomaly_flag":           anomaly_flag[i],
            })

    df = pd.DataFrame(records)

    # ── Introduce realistic missing values (~2%) ───────────────────────────────
    for col in ["oil_rate_bbl_day", "wellhead_pressure_psi", "water_cut_pct"]:
        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    output_path = "data/raw_production_data.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset generated: {output_path}")
    print(f"   Shape       : {df.shape}")
    print(f"   Wells       : {df['well_id'].nunique()}")
    print(f"   Date Range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Basins      : {df['basin'].unique().tolist()}")
    print(f"   Missing vals: {df.isnull().sum().sum()} cells (~2% intentional)")
    print(f"   Anomalies   : {df['anomaly_flag'].sum()} injected events")
    print("\nFirst 5 rows:")
    print(df.head())
