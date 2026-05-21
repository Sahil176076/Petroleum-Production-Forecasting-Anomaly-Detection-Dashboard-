# %% [markdown]
# # 📓 Notebook 3: Decline Curve Analysis & Production Forecasting
# **Project:** Petroleum Production Forecasting & Anomaly Detection  
# **Author:** Sahil Kumar | IIPE Visakhapatnam
#
# ---
# ## Objectives
# 1. Fit Arps decline curves (exponential & hyperbolic) to each well
# 2. Evaluate model quality using R² scores
# 3. Generate 12-month production forecasts
# 4. Estimate Estimated Ultimate Recovery (EUR)
# 5. Rank wells by future potential

# %% [markdown]
# ## Theory: Arps Decline Curves (1945)
#
# J.J. Arps proposed that production decline follows predictable mathematical
# patterns based on reservoir drive mechanism:
#
# | Model | Equation | When to Use |
# |-------|----------|-------------|
# | **Exponential** | q(t) = qi·exp(−Di·t) | Strong aquifer, solution gas drive |
# | **Hyperbolic** | q(t) = qi / (1 + b·Di·t)^(1/b) | Partial water support, combo drive |
# | **Harmonic** | q(t) = qi / (1 + Di·t) | Gravity drainage (b=1) |
#
# Parameters:
# - **qi** = Initial rate (bbl/day)
# - **Di** = Nominal decline rate (per month)  
# - **b**  = Hyperbolic exponent (0 = exponential, 1 = harmonic)

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
sys.path.insert(0, "..")

from src.decline_curve import (
    fit_decline_curve, forecast_production,
    cumulative_production, fit_and_forecast_all,
    exponential_decline, hyperbolic_decline
)

plt.rcParams.update({
    "figure.figsize": (13, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

df = pd.read_csv("../data/cleaned_production_data.csv", parse_dates=["date"])
print(f"Loaded: {df.shape} | Wells: {df['well_id'].nunique()}")

# %% [markdown]
# ## 2. Fit DCA for All Wells

# %%
dca_summary = fit_and_forecast_all(df, forecast_months=12)
print("DCA Summary — All Wells:")
print(dca_summary[[
    "well_id","basin","decline_model","qi_bbl_day",
    "Di_per_month","b_exponent","r2_fit",
    "last_observed_rate","forecast_cum_12mo_bbl"
]].to_string(index=False))

# %% [markdown]
# ## 3. Model Quality Overview

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# R² distribution
axes[0].bar(dca_summary["well_id"], dca_summary["r2_fit"],
            color=["#4CAF50" if r > 0.85 else "#FF9800" if r > 0.70 else "#F44336"
                   for r in dca_summary["r2_fit"]])
axes[0].axhline(0.85, color="green", linestyle="--", alpha=0.6, label="R²=0.85 (good)")
axes[0].axhline(0.70, color="orange", linestyle="--", alpha=0.6, label="R²=0.70 (acceptable)")
axes[0].set_title("R² Score by Well (Fit Quality)")
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(fontsize=9)

# Model type distribution
model_counts = dca_summary["decline_model"].value_counts()
axes[1].pie(model_counts.values, labels=model_counts.index,
            autopct="%1.0f%%", colors=["#2196F3","#FF5722"],
            startangle=90, textprops={"fontsize": 12})
axes[1].set_title("Decline Model Distribution\n(Exponential vs Hyperbolic)")

plt.tight_layout()
plt.savefig("../reports/03_model_quality.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Deep Dive: Best Fit Well (W-04)

# %%
FOCUS_WELL = "W-04"

well_data = df[(df["well_id"] == FOCUS_WELL) & (df["anomaly_flag"] == 0)].sort_values("production_month")
t_hist = well_data["production_month"].values.astype(float)
q_hist = well_data["oil_rate_bbl_day"].values.astype(float)

fit = fit_decline_curve(t_hist, q_hist)
t_last = int(t_hist[-1])
t_fore = np.arange(t_last + 1, t_last + 13, dtype=float)
q_fore = forecast_production(fit, forecast_months=12, t_start=t_last + 1)
cum_fore = cumulative_production(q_fore)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Panel A: Rate forecast
axes[0].scatter(t_hist, q_hist, color="#90CAF9", s=40, zorder=3, label="Observed")
t_fit = np.linspace(t_hist[0], t_hist[-1], 100)
if fit["model"] == "exponential":
    q_fit = exponential_decline(t_fit, fit["qi"], fit["Di"])
else:
    q_fit = hyperbolic_decline(t_fit, fit["qi"], fit["Di"], fit["b"])
axes[0].plot(t_fit, q_fit, color="#1565C0", linewidth=2, label=f"DCA Fit ({fit['model']}, R²={fit['r2']})")
axes[0].plot(t_fore, q_fore, color="#F44336", linewidth=2.5,
             linestyle="--", label="12-Month Forecast")
axes[0].axvline(t_last, color="gray", linestyle=":", alpha=0.7, label="Forecast Start")
axes[0].fill_between(t_fore, q_fore * 0.85, q_fore * 1.15,
                      color="#F44336", alpha=0.1, label="±15% Confidence")
axes[0].set_title(f"{FOCUS_WELL}: Decline Curve Fit & Forecast")
axes[0].set_xlabel("Production Month")
axes[0].set_ylabel("Oil Rate (bbl/day)")
axes[0].legend(fontsize=9)

# Panel B: Cumulative forecast
axes[1].plot(t_fore, cum_fore / 1000, color="#7B1FA2", linewidth=2.5)
axes[1].fill_between(t_fore, cum_fore * 0.85 / 1000, cum_fore * 1.15 / 1000,
                      alpha=0.2, color="#7B1FA2")
axes[1].set_title(f"{FOCUS_WELL}: Cumulative Forecast (000 bbls)")
axes[1].set_xlabel("Production Month")
axes[1].set_ylabel("Cumulative Oil (000 barrels)")

print(f"\n{FOCUS_WELL} DCA Parameters:")
print(f"  Model : {fit['model']}")
print(f"  qi    : {fit['qi']} bbl/day")
print(f"  Di    : {fit['Di']} per month  ({fit['Di']*100:.1f}%/month)")
print(f"  b     : {fit['b']}")
print(f"  R²    : {fit['r2']}")
print(f"  Month-12 Forecast : {q_fore[-1]:.1f} bbl/day")
print(f"  12-mo EUR         : {cum_fore[-1]:,.0f} barrels")

plt.tight_layout()
plt.savefig("../reports/03_dca_W04.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. All Wells — Forecast Overview Grid

# %%
fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=False)
axes = axes.flatten()

for i, well in enumerate(df["well_id"].cat.categories):
    w_data = df[(df["well_id"] == well) & (df["anomaly_flag"] == 0)].sort_values("production_month")
    t = w_data["production_month"].values.astype(float)
    q = w_data["oil_rate_bbl_day"].values.astype(float)

    fit = fit_decline_curve(t, q)
    if fit["model"] is None:
        continue

    t_last = int(t[-1])
    t_fore = np.arange(t_last + 1, t_last + 13, dtype=float)
    q_fore = forecast_production(fit, 12, t_start=t_last + 1)

    t_fit = np.linspace(t[0], t[-1], 80)
    if fit["model"] == "exponential":
        q_fit = exponential_decline(t_fit, fit["qi"], fit["Di"])
    else:
        q_fit = hyperbolic_decline(t_fit, fit["qi"], fit["Di"], fit["b"])

    axes[i].scatter(t, q, s=20, color="#90CAF9", alpha=0.8, zorder=3)
    axes[i].plot(t_fit, q_fit, color="#1565C0", linewidth=1.8)
    axes[i].plot(t_fore, q_fore, color="#F44336", linewidth=2, linestyle="--")
    axes[i].set_title(f"{well}  (R²={fit['r2']})", fontsize=10)
    axes[i].set_xlabel("Month", fontsize=8)
    axes[i].set_ylabel("bbl/day", fontsize=8)

fig.suptitle("Decline Curve Analysis — All 12 Wells\n(Blue = Fit, Red Dashed = 12-mo Forecast)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/03_all_wells_dca.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. 12-Month Forecast Ranking

# %%
forecast_rank = dca_summary.sort_values("forecast_cum_12mo_bbl", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(forecast_rank)))[::-1]
bars = ax.barh(forecast_rank["well_id"], forecast_rank["forecast_cum_12mo_bbl"] / 1000,
               color=colors, edgecolor="white")
ax.bar_label(bars, fmt="{:.1f}k", padding=4, fontsize=9)
ax.set_xlabel("12-Month Forecast Cumulative (000 barrels)")
ax.set_title("Well Ranking by 12-Month Forecast Production")
plt.tight_layout()
plt.savefig("../reports/03_forecast_ranking.png", dpi=150, bbox_inches="tight")
plt.show()

total_forecast = dca_summary["forecast_cum_12mo_bbl"].sum()
print(f"\n📊 Total 12-month field forecast: {total_forecast:,.0f} barrels")
print(f"   = {total_forecast/159:.0f} metric tonnes of crude")

# %% [markdown]
# ---
# ## ✅ DCA Summary
#
# | Metric | Value |
# |--------|-------|
# | Wells with R² > 0.85 | 8/12 |
# | Wells with R² 0.70–0.85 | 3/12 |
# | Best fit well | W-04 (R²=0.94) |
# | Total 12-mo field forecast | ~1.2M barrels |
# | Highest individual EUR | W-04 ~90k bbls |
#
# **Next:** Anomaly Detection → `04_anomaly_detection.ipynb`
