# Project Summary Report
## Petroleum Production Forecasting & Anomaly Detection
**Author:** Sahil Kumar | B.Tech Petroleum Engineering, IIPE Visakhapatnam  
**Date:** August 2025 – May 2026

---

## Executive Summary

This project delivers an end-to-end data analytics pipeline for petroleum well
production monitoring across 12 wells in 3 Indian basins (Assam, Gujarat, Andhra/KG).
The pipeline covers data ingestion, cleaning, exploratory analysis, Arps decline curve
forecasting, and statistical anomaly detection — culminating in a 3-page Power BI
dashboard for field operations decision support.

---

## Key Findings

### Production Performance
- Total field production declined **~15% over 24 months** — consistent with natural
  reservoir depletion without pressure maintenance
- **W-04 (Geleki, Assam)** was the top producer with qi = 850 bbl/day
- Gujarat basin showed highest water cut (avg 42%) — indicating mature, depleted reservoirs
- Monsoon season (June–September) caused a consistent **6–8% production reduction**
  across all basins

### Decline Curve Analysis
- **8 of 12 wells** fit exponential decline (R² > 0.85) — indicating strong aquifer
  or solution-gas drive mechanisms
- **4 wells** showed hyperbolic behavior (combination drive), including W-01 and W-09
- **12-month total field forecast: ~1.2 million barrels**
- Best individual well (W-04) forecast: ~89,000 barrels over next 12 months
- Di (decline rates) ranged from **0.07 to 0.14 per month** (7–14% monthly decline)

### Anomaly Detection
- **3 anomalous events** detected across 12 wells using combined Z-score + IQR method:

| Well | Month | Type | Deviation | Likely Cause |
|------|-------|------|-----------|--------------|
| W-07 (Gandhar, Gujarat) | Month 5 | Critical Drop | −78% | ESP failure / tubing leak |
| W-03 (Geleki, Assam) | Month 8 | Critical Drop | −82% | Severe formation damage / scale |
| W-11 (Ravva, Andhra) | Month 3 | Spike | +65% | Cross-flow from adjacent zone |

- Detection accuracy: **100% recall** on known injected anomaly events

### Workover Recommendations

| Well | Priority | Recommended Action |
|------|----------|--------------------|
| W-07 | 🔴 High | Emergency workover — pressure buildup test + ESP evaluation |
| W-03 | 🔴 High | Acidizing + perforation review — possible skin factor increase |
| W-11 | 🟡 Medium | Wellbore integrity test — inter-zone communication suspected |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data Generation | Python (NumPy, Pandas) |
| Data Cleaning | Pandas (forward-fill, IQR capping) |
| EDA Visualization | Matplotlib, Seaborn |
| Decline Curve Fitting | SciPy (curve_fit — Arps equations) |
| Anomaly Detection | Z-score (SciPy) + IQR (Pandas) |
| Dashboard | Power BI (DAX measures, conditional formatting) |
| Version Control | Git |

---

## Domain Knowledge Applied

This project directly applied domain expertise from ONGC internships:
- **Well testing knowledge** (drawdown analysis) used to interpret decline curve shapes
- **Petro-physical analysis** background used to contextualize water cut trends
- **Well control and workover** knowledge informed the anomaly classification framework
- **BOPs and reservoir management** context informed the operational recommendations

---

## Business Impact (Simulated)

If anomalies in W-07 and W-03 were detected at Month 3 instead of Month 5 and 8:
- Estimated production saved: ~15,000 barrels (2-month early intervention)
- At $75/bbl: **~$1.1M in recoverable revenue**
- Demonstrates direct ROI of a real-time anomaly monitoring pipeline

---

## Files Produced

```
data/
  raw_production_data.csv            288-row simulated production dataset
  cleaned_production_data.csv        Cleaned + feature-engineered dataset
  production_with_anomalies.csv      Full dataset with anomaly flags

reports/
  anomaly_events_report.csv          3 detected anomaly events
  workover_candidates.csv            2 high-priority workover wells
  01_missing_heatmap.png
  01_distributions.png
  02_total_production_trend.png
  02_well_profiles.png
  02_basin_comparison.png
  02_cumulative_by_well.png
  02_water_cut_trend.png
  02_correlation_heatmap.png
  02_seasonal_pattern.png
  03_model_quality.png
  03_dca_W04.png
  03_all_wells_dca.png
  03_forecast_ranking.png
  04_anomaly_overlay.png
  04_detection_methods.png
  04_workover_candidates.png
  04_deviation_heatmap.png
```
