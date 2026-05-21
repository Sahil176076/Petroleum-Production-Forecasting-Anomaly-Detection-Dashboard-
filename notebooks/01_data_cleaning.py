# %% [markdown]
# # 📓 Notebook 1: Data Ingestion & Cleaning
# **Project:** Petroleum Production Forecasting & Anomaly Detection  
# **Author:** Sahil Kumar | IIPE Visakhapatnam  
# **Dataset:** Synthetic EIA-format petroleum production data (12 wells, 24 months)
#
# ---
# ## Objectives
# 1. Load raw production data and inspect structure
# 2. Identify and handle missing values
# 3. Detect and cap outliers
# 4. Engineer useful time-based and derived features
# 5. Export clean dataset for downstream analysis

# %% [markdown]
# ## 1. Imports & Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys

sys.path.insert(0, "..")  # Allow importing from src/

# Visual settings
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
sns.set_palette("tab10")

print("Libraries loaded ✅")
print(f"Pandas  : {pd.__version__}")
print(f"NumPy   : {np.__version__}")

# %% [markdown]
# ## 2. Generate & Load Raw Data
# We first run the data generator to create `raw_production_data.csv`,
# then load it for inspection.

# %%
# Run generator if data doesn't exist
if not os.path.exists("../data/raw_production_data.csv"):
    os.chdir("..")
    os.system("python src/data_generator.py")
    os.chdir("notebooks")

df_raw = pd.read_csv("../data/raw_production_data.csv", parse_dates=["date"])
print(f"Shape        : {df_raw.shape}")
print(f"Date range   : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
print(f"Wells        : {df_raw['well_id'].nunique()} ({df_raw['well_id'].unique().tolist()})")
print(f"Basins       : {df_raw['basin'].unique().tolist()}")
df_raw.head(8)

# %%
# Data types and memory usage
df_raw.info()

# %% [markdown]
# ## 3. Missing Value Analysis

# %%
# Count and visualize missing values
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]
print(missing_df)

# Heatmap of missing values (sample first 100 rows for readability)
fig, ax = plt.subplots(figsize=(10, 4))
sample = df_raw[["oil_rate_bbl_day", "water_cut_pct", "wellhead_pressure_psi"]].head(100)
sns.heatmap(sample.isnull(), cbar=False, yticklabels=False,
            cmap="Reds", ax=ax)
ax.set_title("Missing Values in First 100 Rows (Red = Missing)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/01_missing_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Observation:** ~2% missing across 3 numeric columns — introduced by sensor dropout
# and communication failures common in field telemetry systems.

# %% [markdown]
# ## 4. Clean the Data

# %%
from src.cleaning import load_and_clean

df_clean = load_and_clean(
    filepath="../data/raw_production_data.csv",
    output_path="../data/cleaned_production_data.csv"
)
print(f"\nCleaned shape: {df_clean.shape}")
df_clean.head()

# %% [markdown]
# ## 5. Before vs After: Missing Value Comparison

# %%
print("BEFORE cleaning:")
print(df_raw[["oil_rate_bbl_day","water_cut_pct","wellhead_pressure_psi"]].isnull().sum())
print("\nAFTER cleaning:")
print(df_clean[["oil_rate_bbl_day","water_cut_pct","wellhead_pressure_psi"]].isnull().sum())

# %% [markdown]
# ## 6. Distribution of Key Variables

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df_clean["oil_rate_bbl_day"].hist(bins=30, ax=axes[0], color="#2196F3", edgecolor="white")
axes[0].set_title("Oil Rate Distribution (bbl/day)")
axes[0].set_xlabel("bbl/day")

df_clean["water_cut_pct"].hist(bins=30, ax=axes[1], color="#FF5722", edgecolor="white")
axes[1].set_title("Water Cut Distribution (%)")
axes[1].set_xlabel("%")

df_clean["wellhead_pressure_psi"].hist(bins=30, ax=axes[2], color="#4CAF50", edgecolor="white")
axes[2].set_title("Wellhead Pressure Distribution (psi)")
axes[2].set_xlabel("psi")

for ax in axes:
    ax.set_ylabel("Frequency")

plt.suptitle("Distribution of Key Production Parameters", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/01_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Wells per Basin Overview

# %%
basin_summary = df_clean.groupby(["basin", "field"], observed=True).agg(
    wells=("well_id", "nunique"),
    avg_rate=("oil_rate_bbl_day", "mean"),
    total_production_bbl=("oil_rate_bbl_month", "sum")
).reset_index()

print("\nBasin & Field Summary:")
print(basin_summary.round(1).to_string(index=False))

# %% [markdown]
# ## 8. Feature Engineering Review

# %%
# Check new engineered columns
print("Engineered columns added:")
new_cols = ["year","month","quarter","production_month","oil_rate_bbl_month","wc_category"]
print(df_clean[new_cols].head(10).to_string())

# %% [markdown]
# ---
# ## ✅ Summary
#
# | Step | Action | Result |
# |------|--------|--------|
# | Load | Read 288-row raw CSV | 11 columns, 3 basins |
# | Missing | Forward-fill within wells + median fallback | 0 missing values |
# | Outliers | IQR capping (factor=2.5), anomalies preserved | Rates bounded sensibly |
# | Features | Year, month, quarter, wc_category, monthly volume | 6 new columns |
#
# **Next:** Exploratory Data Analysis → `02_eda.ipynb`
