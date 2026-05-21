"""
anomaly_detection.py
====================
Production anomaly detection for petroleum wells using statistical methods.

Two complementary approaches are used together:

1. Z-Score Method (Global):
   Flags a rate as anomalous if it deviates > N standard deviations
   from the well's expected decline trend (residuals from DCA fit).
   Good at catching sudden spikes or drops relative to trend.

2. IQR Method (Rolling Window):
   Uses a rolling 3-month IQR window per well to flag local anomalies.
   Good at catching step-changes that occur gradually.

Classification:
   🔴 CRITICAL DROP  : rate < 40% of expected (well failure / reservoir damage)
   🟠 MODERATE DROP  : rate 40–70% of expected (partial blockage / scale)
   🟡 SPIKE          : rate > 160% of expected (cross-flow / measurement error)
   ✅ NORMAL          : within expected range

Domain context:
   In ONGC operations, anomalies trigger:
   - Field visit for visual inspection
   - Pressure buildup test to assess skin damage
   - Workover evaluation (ESP replacement, acidizing, perforation)

Usage:
    from src.anomaly_detection import detect_anomalies
    df_with_anomalies = detect_anomalies(df_clean)
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")


# ── Thresholds ─────────────────────────────────────────────────────────────────
Z_THRESHOLD        = 2.5    # Z-score cutoff for residual anomaly
IQR_FACTOR         = 2.0    # IQR multiplier for rolling window
CRITICAL_DROP_PCT  = 0.40   # < 40% of expected = critical
MODERATE_DROP_PCT  = 0.70   # < 70% of expected = moderate
SPIKE_PCT          = 1.60   # > 160% of expected = spike


def compute_rolling_baseline(group: pd.DataFrame,
                              window: int = 3) -> pd.Series:
    """
    Compute a rolling median of oil_rate_bbl_day as the local baseline.

    Rolling median is robust to outliers (unlike rolling mean), making
    it suitable as a reference for anomaly comparison.

    Args:
        group  : DataFrame slice for one well, sorted by date
        window : Rolling window size in months (default 3)

    Returns:
        Series of rolling median values (same index as group)
    """
    return (
        group["oil_rate_bbl_day"]
        .rolling(window=window, min_periods=1, center=False)
        .median()
        .shift(1)   # Shift to avoid look-ahead: baseline is from previous months
        .bfill()
    )


def zscore_anomaly(series: pd.Series, threshold: float = Z_THRESHOLD) -> pd.Series:
    """
    Flag anomalies using Z-score on residuals (observed − rolling baseline).

    A high absolute Z-score means the rate is far from its local trend.

    Args:
        series    : Residual series (observed − baseline)
        threshold : Z-score cutoff

    Returns:
        Boolean Series — True where anomaly detected
    """
    if series.std() == 0:
        return pd.Series(False, index=series.index)
    z = np.abs((series - series.mean()) / series.std())
    return z > threshold


def iqr_anomaly(series: pd.Series, factor: float = IQR_FACTOR) -> pd.Series:
    """
    Flag anomalies using IQR (Interquartile Range) fences.

    Lower fence = Q1 - factor * IQR
    Upper fence = Q3 + factor * IQR

    Args:
        series : Production rate series
        factor : IQR multiplier (default 2.0 — permissive)

    Returns:
        Boolean Series — True where anomaly detected
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def classify_anomaly(observed: float, baseline: float) -> str:
    """
    Classify the type and severity of an anomaly.

    Args:
        observed : Actual production rate
        baseline : Expected (rolling baseline) rate

    Returns:
        String label: "Critical Drop", "Moderate Drop", "Spike", or "Normal"
    """
    if baseline <= 0:
        return "Normal"
    ratio = observed / baseline
    if ratio < CRITICAL_DROP_PCT:
        return "Critical Drop"
    elif ratio < MODERATE_DROP_PCT:
        return "Moderate Drop"
    elif ratio > SPIKE_PCT:
        return "Spike"
    else:
        return "Normal"


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main anomaly detection pipeline.

    Steps:
      1. Sort by well and date
      2. Compute rolling 3-month median baseline per well
      3. Calculate residuals (observed − baseline)
      4. Apply Z-score test on residuals
      5. Apply IQR test on raw rates
      6. Flag any row where EITHER test triggers
      7. Classify anomaly type and severity
      8. Compute deviation percentage for reporting

    Args:
        df : Cleaned production DataFrame (from cleaning.py)

    Returns:
        Original DataFrame with added columns:
            rolling_baseline_bbl_day  : Expected rate from rolling median
            residual                  : observed − baseline
            zscore_flag               : Boolean — Z-score anomaly
            iqr_flag                  : Boolean — IQR anomaly
            anomaly_detected          : Boolean — either method triggered
            anomaly_type              : "Critical Drop" / "Moderate Drop" / "Spike" / "Normal"
            deviation_pct             : % deviation from baseline (signed)
    """
    df = df.sort_values(["well_id", "date"]).copy()
    results = []

    for well_id, group in df.groupby("well_id", observed=True):
        group = group.copy().reset_index(drop=True)

        # ── Step 1: Rolling baseline ─────────────────────────────────────────
        baseline = compute_rolling_baseline(group)
        group["rolling_baseline_bbl_day"] = baseline.values

        # ── Step 2: Residuals ────────────────────────────────────────────────
        group["residual"] = group["oil_rate_bbl_day"] - group["rolling_baseline_bbl_day"]

        # ── Step 3: Z-score on residuals ─────────────────────────────────────
        group["zscore_flag"] = zscore_anomaly(group["residual"]).values

        # ── Step 4: IQR on rates ─────────────────────────────────────────────
        group["iqr_flag"] = iqr_anomaly(group["oil_rate_bbl_day"]).values

        # ── Step 5: Combined flag ─────────────────────────────────────────────
        group["anomaly_detected"] = group["zscore_flag"] | group["iqr_flag"]

        # ── Step 6: Classify ─────────────────────────────────────────────────
        group["anomaly_type"] = group.apply(
            lambda row: classify_anomaly(
                row["oil_rate_bbl_day"],
                row["rolling_baseline_bbl_day"]
            ),
            axis=1
        )

        # ── Step 7: Deviation % ───────────────────────────────────────────────
        group["deviation_pct"] = (
            (group["oil_rate_bbl_day"] - group["rolling_baseline_bbl_day"])
            / group["rolling_baseline_bbl_day"].replace(0, np.nan)
            * 100
        ).round(1)

        results.append(group)

    df_out = pd.concat(results, ignore_index=True)
    return df_out


def anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a clean anomaly report for detected events.

    Args:
        df : DataFrame output from detect_anomalies()

    Returns:
        Filtered DataFrame of only anomalous rows with key columns
    """
    cols = [
        "date", "well_id", "basin", "field",
        "oil_rate_bbl_day", "rolling_baseline_bbl_day",
        "deviation_pct", "anomaly_type",
        "zscore_flag", "iqr_flag"
    ]
    anomalies = df[df["anomaly_detected"]].copy()
    # Remove normal-classified rows (edge cases where flag fires but ratio is ok)
    anomalies = anomalies[anomalies["anomaly_type"] != "Normal"]
    return anomalies[cols].sort_values("deviation_pct")


def workover_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify wells that are likely workover candidates.

    Criteria:
      - Has at least 1 Critical Drop anomaly, OR
      - Has 2+ Moderate Drop anomalies in the history

    Args:
        df : DataFrame output from detect_anomalies()

    Returns:
        DataFrame of wells with workover priority score
    """
    anom = df[df["anomaly_detected"] & (df["anomaly_type"] != "Normal")].copy()

    critical = anom[anom["anomaly_type"] == "Critical Drop"].groupby(
        "well_id", observed=True
    ).size().rename("critical_events")

    moderate = anom[anom["anomaly_type"] == "Moderate Drop"].groupby(
        "well_id", observed=True
    ).size().rename("moderate_events")

    summary = pd.concat([critical, moderate], axis=1).fillna(0).astype(int)
    summary["priority_score"] = summary["critical_events"] * 3 + summary["moderate_events"]
    summary = summary[summary["priority_score"] > 0].sort_values(
        "priority_score", ascending=False
    )

    # Add last known rate
    last_rate = df.sort_values("date").groupby(
        "well_id", observed=True
    )["oil_rate_bbl_day"].last().rename("last_rate_bbl_day")
    summary = summary.join(last_rate, how="left")

    summary["workover_priority"] = pd.cut(
        summary["priority_score"],
        bins=[0, 2, 5, 100],
        labels=["Low", "Medium", "High"]
    )
    return summary.reset_index()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.cleaning import load_and_clean

    df = load_and_clean("data/raw_production_data.csv")
    df_anom = detect_anomalies(df)

    print("\n" + "=" * 65)
    print(" ANOMALY DETECTION RESULTS")
    print("=" * 65)
    summary = anomaly_summary(df_anom)
    print(f"\nTotal anomalies detected: {len(summary)}")
    print(summary.to_string(index=False))

    print("\n" + "=" * 65)
    print(" WORKOVER CANDIDATES")
    print("=" * 65)
    candidates = workover_candidates(df_anom)
    print(candidates.to_string(index=False))
