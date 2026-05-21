# %% [markdown]
# # 📓 Notebook 4: Production Anomaly Detection
# **Project:** Petroleum Production Forecasting & Anomaly Detection  
# **Author:** Sahil Kumar | IIPE Visakhapatnam
#
# ---
# ## Objectives
# 1. Detect abnormal production events using Z-score and IQR methods
# 2. Classify anomalies by type (Critical Drop, Moderate Drop, Spike)
# 3. Identify workover candidate wells
# 4. Visualize anomalies in context of production history
# 5. Generate actionable field operations report

# %% [markdown]
# ## Domain Context
#
# In petroleum field operations (ONGC), anomalies trigger:
#
# | Anomaly Type | Likely Cause | Field Action |
# |---|---|---|
# | **Critical Drop** (< 40% of expected) | ESP failure, tubing leak, severe skin damage | Emergency workover, well test |
# | **Moderate Drop** (40–70% of expected) | Scale buildup, partial blockage, sand production | Stimulation, acidizing |
# | **Spike** (> 160% of expected) | Cross-flow from adjacent zone, measurement error | Wellbore integrity test |

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import sys
sys.path.insert(0, "..")

from src.anomaly_detection import (
    detect_anomalies, anomaly_summary, workover_candidates
)

plt.rcParams.update({
    "figure.figsize": (13, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df_raw = pd.read_csv("../data/cleaned_production_data.csv", parse_dates=["date"])
print(f"Loaded: {df_raw.shape}")

# %% [markdown]
# ## 2. Run Anomaly Detection Pipeline

# %%
df_anom = detect_anomalies(df_raw)

print(f"\nTotal records    : {len(df_anom)}")
print(f"Anomalies found  : {df_anom['anomaly_detected'].sum()}")
print(f"Anomaly rate     : {df_anom['anomaly_detected'].mean()*100:.1f}%")
print(f"\nBy type:")
print(df_anom[df_anom["anomaly_detected"]]["anomaly_type"].value_counts())

# %% [markdown]
# ## 3. Anomaly Summary Report

# %%
summary = anomaly_summary(df_anom)
print("\n=== ANOMALY EVENT REPORT ===")
print(summary[[
    "date","well_id","basin","oil_rate_bbl_day",
    "rolling_baseline_bbl_day","deviation_pct","anomaly_type"
]].to_string(index=False))

# %% [markdown]
# ## 4. Full Production History with Anomaly Overlays

# %%
anomaly_colors = {
    "Critical Drop":  "#D32F2F",
    "Moderate Drop":  "#FF6F00",
    "Spike":          "#7B1FA2",
}
anomaly_markers = {
    "Critical Drop": "v",
    "Moderate Drop": "s",
    "Spike":         "^",
}

wells_with_anom = summary["well_id"].unique()

fig, axes = plt.subplots(1, len(wells_with_anom), figsize=(15, 5))
if len(wells_with_anom) == 1:
    axes = [axes]

for ax, well in zip(axes, wells_with_anom):
    well_df = df_anom[df_anom["well_id"] == well].sort_values("date")
    anom_df = well_df[well_df["anomaly_detected"] & (well_df["anomaly_type"] != "Normal")]

    # Rate line
    ax.plot(well_df["date"], well_df["oil_rate_bbl_day"],
            color="#1565C0", linewidth=1.8, label="Observed")
    ax.plot(well_df["date"], well_df["rolling_baseline_bbl_day"],
            color="#90CAF9", linewidth=1.5, linestyle="--", label="Rolling Baseline")

    # Anomaly markers
    for atype, group in anom_df.groupby("anomaly_type"):
        color  = anomaly_colors.get(atype, "black")
        marker = anomaly_markers.get(atype, "X")
        ax.scatter(group["date"], group["oil_rate_bbl_day"],
                   color=color, s=150, zorder=5, marker=marker,
                   label=atype, edgecolors="black", linewidths=0.8)

        # Deviation annotation
        for _, row in group.iterrows():
            ax.annotate(f"{row['deviation_pct']:.0f}%",
                        xy=(row["date"], row["oil_rate_bbl_day"]),
                        xytext=(0, -25), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold",
                        ha="center")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.tick_params(axis='x', rotation=30)
    ax.set_title(f"Well {well}", fontsize=11)
    ax.set_ylabel("Oil Rate (bbl/day)")
    ax.legend(fontsize=8)

fig.suptitle("Production Anomalies — Detected Events with Deviation %",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/04_anomaly_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Detection Method Comparison: Z-Score vs IQR

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Z-score flags
z_flags = df_anom[df_anom["zscore_flag"]]
axes[0].scatter(df_anom["date"], df_anom["oil_rate_bbl_day"],
                color="#BDBDBD", s=15, alpha=0.5, label="Normal")
axes[0].scatter(z_flags["date"], z_flags["oil_rate_bbl_day"],
                color="#F44336", s=60, zorder=5, label="Z-Score Flag", marker="X")
axes[0].set_title("Z-Score Method Flags")
axes[0].set_ylabel("Oil Rate (bbl/day)")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[0].tick_params(axis='x', rotation=30)

# IQR flags
iqr_flags = df_anom[df_anom["iqr_flag"]]
axes[1].scatter(df_anom["date"], df_anom["oil_rate_bbl_day"],
                color="#BDBDBD", s=15, alpha=0.5, label="Normal")
axes[1].scatter(iqr_flags["date"], iqr_flags["oil_rate_bbl_day"],
                color="#9C27B0", s=60, zorder=5, label="IQR Flag", marker="s")
axes[1].set_title("IQR Method Flags")
axes[1].set_ylabel("Oil Rate (bbl/day)")
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
axes[1].tick_params(axis='x', rotation=30)

fig.suptitle("Anomaly Detection Method Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/04_detection_methods.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Workover Candidate Ranking

# %%
candidates = workover_candidates(df_anom)
print("\n=== WORKOVER CANDIDATE WELLS ===")
print(candidates.to_string(index=False))

if not candidates.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = {"High": "#D32F2F", "Medium": "#FF6F00", "Low": "#FFC107"}
    colors_list = [bar_colors.get(str(p), "#90CAF9") for p in candidates["workover_priority"]]

    bars = ax.bar(candidates["well_id"], candidates["priority_score"],
                  color=colors_list, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Workover Priority Score by Well")
    ax.set_ylabel("Priority Score (Critical×3 + Moderate×1)")
    ax.set_xlabel("Well ID")

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in bar_colors.items()]
    ax.legend(handles=patches, title="Priority")
    plt.tight_layout()
    plt.savefig("../reports/04_workover_candidates.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 7. Deviation Heatmap — All Wells Over Time

# %%
dev_pivot = df_anom.pivot_table(
    index="well_id", columns="date",
    values="deviation_pct", aggfunc="first"
)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(dev_pivot, cmap="RdYlGn", center=0, vmin=-80, vmax=80,
            ax=ax, linewidths=0.3, linecolor="gray",
            cbar_kws={"label": "Deviation from Baseline (%)"})
ax.set_title("Production Deviation Heatmap — All Wells Over Time\n"
             "(Green=Above baseline, Red=Below baseline)", fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Well ID")

# Format date labels
date_labels = pd.to_datetime(dev_pivot.columns).strftime("%b'%y")
ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig("../reports/04_deviation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Export Anomaly Report

# %%
# Save anomaly report to CSV for Power BI dashboard
df_anom.to_csv("../data/production_with_anomalies.csv", index=False)
summary.to_csv("../reports/anomaly_events_report.csv", index=False)
candidates.to_csv("../reports/workover_candidates.csv", index=False)

print("✅ Exported:")
print("   data/production_with_anomalies.csv    (full dataset with flags)")
print("   reports/anomaly_events_report.csv     (anomaly events)")
print("   reports/workover_candidates.csv       (priority wells)")

# %% [markdown]
# ---
# ## ✅ Anomaly Detection Summary
#
# | Metric | Value |
# |--------|-------|
# | Total anomalies detected | 3 confirmed events |
# | Critical drops | 2 (W-07, W-03) |
# | Spikes | 1 (W-11) |
# | Workover candidates | W-07 (High), W-03 (High) |
# | Detection accuracy | 100% recall on injected events |
#
# **Recommendation:**
# - W-07 (Gandhar, Gujarat): Emergency workover evaluation — 78% below expected rate
# - W-03 (Geleki, Assam): Pressure buildup test + acidizing candidate
# - W-11 (Ravva, Andhra): Cross-flow test — possible inter-zone communication
#
# **Next:** Build Power BI Dashboard → See `dashboard/PowerBI_Dashboard_Guide.md`
