# Technical Methodology
## Petroleum Production Forecasting & Anomaly Detection

---

## 1. Data Generation Methodology

### Why Synthetic Data?
Real ONGC production data is classified. We use synthetic data modeled after
publicly available EIA (U.S. Energy Information Administration) format, with
parameters calibrated to realistic Indian basin production ranges.

### Realism Parameters
| Parameter | Range Used | Real-World Range |
|-----------|-----------|-----------------|
| Initial Rate (qi) | 290–850 bbl/day | 100–2000 bbl/day |
| Monthly Decline (Di) | 7–14% | 5–20% |
| Water Cut (initial) | 12–25% | 5–40% |
| Wellhead Pressure | 800–1500 psi | 500–3000 psi |

---

## 2. Data Cleaning Methodology

### Missing Value Strategy
**Problem:** Field telemetry systems experience ~2% data loss (sensor dropout,
communication failures, planned shutdowns).

**Solution — Two-step approach:**
1. **Forward-fill within well** (`ffill`): Uses the last known value. Physically
   justified — production rate doesn't instantly jump between measurements.
2. **Backward-fill** for start-of-record gaps, then **median fallback** for edge cases.

**Why not mean imputation?** Mean imputation biases the decline curve — it pulls
early (high) and late (low) values toward center, flattening the natural decline.

### Outlier Capping
**Method:** IQR with factor 2.5 (less aggressive than standard 1.5).

We use factor 2.5 because petroleum production has naturally high variance —
using 1.5 would cap legitimate operational fluctuations. Anomaly-flagged rows
are **excluded** from capping (preserved for detection).

---

## 3. Decline Curve Analysis Methodology

### Arps Equation Background
Arps (1945) showed empirically that production decline can be described by three
parameters: qi (initial rate), Di (decline rate), and b (shape parameter).

**Physical interpretation of b:**
- b = 0: Exponential — pressure depletes at rate proportional to production
         (solution gas drive, strong aquifer)
- b = 0.5: Moderate hyperbolic — combination drive mechanisms
- b → 1: Harmonic — gravity drainage (slowest decline)

### Fitting Procedure
We use **SciPy's `curve_fit`** (non-linear least squares, Levenberg-Marquardt):
1. Set physically meaningful bounds: qi ∈ [0, 2×q₀], Di ∈ [0.001, 1.0], b ∈ [0.01, 0.99]
2. Initialize with p0 = [q[0], 0.05] for exponential, [q[0], 0.05, 0.5] for hyperbolic
3. Fit both models independently
4. Select best by R² (coefficient of determination)

**Anomaly rows excluded from fitting** — a 78% production drop would severely
bias the decline parameters if included.

### Forecast Uncertainty
We apply ±15% confidence bands on forecasts. In practice, P10/P50/P90 Monte Carlo
simulation would be used — beyond scope for this project.

---

## 4. Anomaly Detection Methodology

### Why Two Methods?

| Method | Strength | Weakness |
|--------|----------|----------|
| Z-Score on residuals | Catches sudden deviations from trend | Sensitive to trend model error |
| IQR rolling window | Catches local step-changes | May miss gradual drift |

Using both in OR-combination maximizes recall (catching all true anomalies)
at the cost of slightly higher false positive rate.

### Rolling Baseline
We use a **3-month rolling median** (shifted by 1 to avoid look-ahead) as the
expected rate. Median is preferred over mean — it's robust to the very anomalies
we're trying to detect.

### Z-Score Threshold = 2.5
Standard threshold is 2.0 (95% CI) or 3.0 (99.7% CI).
We use 2.5 as a compromise — in field operations, missing a critical drop
(false negative) is more costly than an unnecessary site visit (false positive).

### Severity Classification
Based on ratio of observed to baseline rate:
```
Critical Drop  : ratio < 0.40  →  Emergency response
Moderate Drop  : ratio 0.40-0.70  →  Scheduled investigation
Spike          : ratio > 1.60  →  Integrity check
Normal         : 0.70 ≤ ratio ≤ 1.60  →  No action
```

These thresholds are industry-informed — a >60% production drop in a single
month almost certainly indicates mechanical failure rather than natural decline.

---

## 5. SQL Aggregation Layer

For the Power BI dashboard data model, key aggregations are pre-computed:

```sql
-- Monthly production by basin
SELECT
    strftime('%Y-%m', date) AS month,
    basin,
    SUM(oil_rate_bbl_day * 30)  AS monthly_oil_bbl,
    AVG(water_cut_pct)          AS avg_wc_pct,
    COUNT(DISTINCT well_id)     AS active_wells
FROM production_with_anomalies
WHERE anomaly_detected = 0   -- exclude anomaly months from aggregation
GROUP BY month, basin
ORDER BY month, basin;

-- Well performance summary
SELECT
    well_id,
    basin,
    field,
    MIN(oil_rate_bbl_day)  AS min_rate,
    MAX(oil_rate_bbl_day)  AS max_rate,
    AVG(oil_rate_bbl_day)  AS avg_rate,
    SUM(oil_rate_bbl_day * 30) AS cumulative_oil_bbl,
    MAX(water_cut_pct)     AS final_wc_pct,
    SUM(anomaly_detected)  AS anomaly_events
FROM production_with_anomalies
GROUP BY well_id, basin, field
ORDER BY cumulative_oil_bbl DESC;
```

---

## 6. Limitations & Future Improvements

| Limitation | Future Improvement |
|------------|-------------------|
| Synthetic data — not real ONGC records | Use EIA open API for real US field data |
| No pressure transient analysis | Integrate buildup test interpretation |
| No multi-phase flow modeling | Add gas-oil-ratio (GOR) and choke modeling |
| Static forecast (no uncertainty) | Monte Carlo P10/P50/P90 simulation |
| No ML-based decline fitting | Test XGBoost/LSTM for non-Arps wells |
| Manual Power BI build | Automate with Python-pptx or Plotly Dash |
