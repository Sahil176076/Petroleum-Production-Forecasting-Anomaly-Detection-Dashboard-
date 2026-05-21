"""
decline_curve.py
================
Arps Decline Curve Analysis (DCA) for petroleum production forecasting.

Implements:
  - Exponential decline fitting  :  q(t) = qi * exp(-Di * t)
  - Hyperbolic decline fitting   :  q(t) = qi / (1 + b*Di*t)^(1/b)
  - Automatic model selection    :  picks best fit by R²
  - 12-month forward forecasting
  - Cumulative production estimation (EUR — Estimated Ultimate Recovery)

Theory:
  Arps (1945) decline curves are the industry standard for production
  forecasting. The three parameters (qi, Di, b) characterize reservoir
  drive mechanism:
    b = 0   → Exponential  (solution gas drive, strong water drive)
    0<b<1   → Hyperbolic   (combination drive, partial water support)
    b = 1   → Harmonic     (gravity drainage — rare)

Usage (as module):
    from src.decline_curve import fit_and_forecast_all
    results = fit_and_forecast_all(df_clean)

Usage (standalone):
    python src/decline_curve.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


# ── Decline Curve Models ───────────────────────────────────────────────────────

def exponential_decline(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """
    Arps Exponential Decline.

    q(t) = qi * exp(-Di * t)

    Args:
        t  : Time (months from production start)
        qi : Initial rate (bbl/day)
        Di : Nominal decline rate (1/month)

    Returns:
        Predicted rate array
    """
    return qi * np.exp(-Di * t)


def hyperbolic_decline(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """
    Arps Hyperbolic Decline.

    q(t) = qi / (1 + b * Di * t)^(1/b)

    Args:
        t  : Time (months)
        qi : Initial rate (bbl/day)
        Di : Nominal decline rate (1/month)
        b  : Hyperbolic exponent (0 < b ≤ 1)

    Returns:
        Predicted rate array
    """
    return qi / np.power(np.maximum(1 + b * Di * t, 1e-6), 1.0 / b)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot

    Args:
        y_true : Observed values
        y_pred : Predicted values

    Returns:
        R² score (float, ideally close to 1.0)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def fit_decline_curve(t: np.ndarray, q: np.ndarray) -> dict:
    """
    Fit both exponential and hyperbolic models to production data.
    Automatically selects the better model by R².

    Args:
        t : Time array (months, 0-based)
        q : Observed production rate array (bbl/day)

    Returns:
        dict with keys:
            model       : "exponential" or "hyperbolic"
            qi, Di, b   : Fitted parameters
            r2          : R² of best fit
            y_fit       : Fitted values on historical data
    """
    result = {"model": None, "qi": None, "Di": None, "b": 0.0, "r2": -np.inf, "y_fit": None}

    # Guard: need at least 4 data points
    if len(t) < 4 or np.std(q) < 1:
        return result

    q = np.array(q, dtype=float)
    t = np.array(t, dtype=float)

    # ── Exponential Fit ────────────────────────────────────────────────────────
    try:
        p0_exp  = [q[0], 0.05]
        bounds_exp = ([0, 0.001], [q[0]*2, 1.0])
        popt_exp, _ = curve_fit(exponential_decline, t, q, p0=p0_exp,
                                 bounds=bounds_exp, maxfev=5000)
        y_exp = exponential_decline(t, *popt_exp)
        r2_exp = r_squared(q, y_exp)
    except Exception:
        r2_exp = -np.inf
        popt_exp = [q[0], 0.1]
        y_exp = np.zeros_like(q)

    # ── Hyperbolic Fit ─────────────────────────────────────────────────────────
    try:
        p0_hyp  = [q[0], 0.05, 0.5]
        bounds_hyp = ([0, 0.001, 0.01], [q[0]*2, 1.0, 0.99])
        popt_hyp, _ = curve_fit(hyperbolic_decline, t, q, p0=p0_hyp,
                                  bounds=bounds_hyp, maxfev=5000)
        y_hyp = hyperbolic_decline(t, *popt_hyp)
        r2_hyp = r_squared(q, y_hyp)
    except Exception:
        r2_hyp = -np.inf
        popt_hyp = [q[0], 0.1, 0.5]
        y_hyp = np.zeros_like(q)

    # ── Model Selection ────────────────────────────────────────────────────────
    if r2_hyp > r2_exp:
        result.update({
            "model": "hyperbolic",
            "qi": round(popt_hyp[0], 2),
            "Di": round(popt_hyp[1], 4),
            "b":  round(popt_hyp[2], 3),
            "r2": round(r2_hyp, 4),
            "y_fit": y_hyp,
        })
    else:
        result.update({
            "model": "exponential",
            "qi": round(popt_exp[0], 2),
            "Di": round(popt_exp[1], 4),
            "b":  0.0,
            "r2": round(r2_exp, 4),
            "y_fit": y_exp,
        })

    return result


def forecast_production(fit: dict, forecast_months: int = 12,
                         t_start: int = 0) -> np.ndarray:
    """
    Generate forward production forecast using fitted decline parameters.

    Args:
        fit            : Output dict from fit_decline_curve()
        forecast_months: Number of months to forecast ahead
        t_start        : Time offset (last historical month index)

    Returns:
        Array of forecasted rates (length = forecast_months)
    """
    t_future = np.arange(t_start, t_start + forecast_months, dtype=float)

    if fit["model"] == "exponential":
        return exponential_decline(t_future, fit["qi"], fit["Di"])
    elif fit["model"] == "hyperbolic":
        return hyperbolic_decline(t_future, fit["qi"], fit["Di"], fit["b"])
    else:
        return np.zeros(forecast_months)


def cumulative_production(rates: np.ndarray, days_per_period: int = 30) -> np.ndarray:
    """
    Calculate cumulative production from rate array.

    Uses trapezoidal approximation:  Np = Σ(q_i × Δt)

    Args:
        rates          : Production rate array (bbl/day)
        days_per_period: Days per time step (default 30 for monthly)

    Returns:
        Cumulative production array (barrels)
    """
    return np.cumsum(rates * days_per_period)


def fit_and_forecast_all(df: pd.DataFrame,
                          forecast_months: int = 12) -> pd.DataFrame:
    """
    Run DCA for every well in the dataset and compile a summary table.

    Args:
        df              : Cleaned production DataFrame
        forecast_months : Months to forecast ahead

    Returns:
        pd.DataFrame with one row per well containing:
            well_id, basin, field, model, qi, Di, b, r2,
            last_observed_rate, forecast_month_12_rate,
            forecast_cum_12mo_bbl, eur_comment
    """
    # Exclude anomaly rows from fitting (they distort the decline model)
    df_fit = df[df["anomaly_flag"] == 0].copy()
    rows = []

    for well_id, group in df_fit.groupby("well_id", observed=True):
        group = group.sort_values("production_month")
        t = group["production_month"].values.astype(float)
        q = group["oil_rate_bbl_day"].values.astype(float)

        fit = fit_decline_curve(t, q)
        if fit["model"] is None:
            continue

        t_last = int(t[-1])
        forecast = forecast_production(fit, forecast_months, t_start=t_last + 1)
        cum_forecast = cumulative_production(forecast)

        meta = group.iloc[0]
        rows.append({
            "well_id":                 well_id,
            "basin":                   meta["basin"],
            "field":                   meta["field"],
            "decline_model":           fit["model"],
            "qi_bbl_day":              fit["qi"],
            "Di_per_month":            fit["Di"],
            "b_exponent":              fit["b"],
            "r2_fit":                  fit["r2"],
            "last_observed_rate":      round(q[-1], 1),
            "forecast_mo12_rate":      round(forecast[-1], 1),
            "forecast_cum_12mo_bbl":   round(cum_forecast[-1], 0),
            "data_points_used":        len(t),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick standalone test
    from src.cleaning import load_and_clean

    df = load_and_clean("data/raw_production_data.csv")
    summary = fit_and_forecast_all(df)

    print("\n" + "=" * 70)
    print(" DECLINE CURVE ANALYSIS — WELL SUMMARY")
    print("=" * 70)
    print(summary[[
        "well_id", "basin", "decline_model", "qi_bbl_day",
        "Di_per_month", "r2_fit", "forecast_cum_12mo_bbl"
    ]].to_string(index=False))

    best_well = summary.loc[summary["forecast_cum_12mo_bbl"].idxmax()]
    print(f"\n🏆 Best well for next 12 months: {best_well['well_id']} "
          f"({best_well['basin']}) — "
          f"{int(best_well['forecast_cum_12mo_bbl']):,} bbls forecast")
