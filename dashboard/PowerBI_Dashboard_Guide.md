# 📊 Power BI Dashboard — Setup Guide
## Petroleum Production Forecasting & Anomaly Detection

---

## Overview

This guide walks through building the 3-page interactive Power BI dashboard
using the CSV outputs generated from the Python analysis notebooks.

**Pages:**
1. **Field Overview** — Total production, basin comparison, KPI cards
2. **Well Performance & Forecast** — DCA results, decline curve chart, ranking
3. **Anomaly & Workover Report** — Anomaly events, deviation heatmap, workover priority

---

## Step 1: Connect Data Sources

### In Power BI Desktop → Home → Get Data → Text/CSV

Import these 3 files:

| File | Use |
|------|-----|
| `data/production_with_anomalies.csv` | Main fact table |
| `data/cleaned_production_data.csv` | Supplementary clean data |
| `reports/anomaly_events_report.csv` | Anomaly detail table |

### Data Model (Star Schema)

```
production_with_anomalies (fact)
    ├── well_id → dim_wells (well_id, basin, field, decline_type)
    └── date    → dim_date  (date, year, month, quarter, month_name)
```

Create `dim_date` table via:
**Modeling → New Table:**
```dax
dim_date = CALENDAR(DATE(2023,1,1), DATE(2024,12,31))
```

---

## Step 2: DAX Measures

Create these in **Modeling → New Measure:**

```dax
-- Total Oil Production (barrels)
Total Oil Production =
    SUMX(production_with_anomalies,
         production_with_anomalies[oil_rate_bbl_day] * 30)

-- Average Daily Rate
Avg Daily Rate =
    AVERAGE(production_with_anomalies[oil_rate_bbl_day])

-- Active Wells
Active Wells =
    DISTINCTCOUNT(production_with_anomalies[well_id])

-- Anomaly Count
Anomaly Count =
    COUNTROWS(FILTER(production_with_anomalies,
              production_with_anomalies[anomaly_detected] = TRUE()))

-- YoY Production Change
YoY Change % =
VAR current_year = MAX(dim_date[Year])
VAR curr =
    CALCULATE([Total Oil Production],
              dim_date[Year] = current_year)
VAR prev =
    CALCULATE([Total Oil Production],
              dim_date[Year] = current_year - 1)
RETURN
    IF(prev = 0, BLANK(), DIVIDE(curr - prev, prev) * 100)

-- Average Water Cut
Avg Water Cut =
    AVERAGE(production_with_anomalies[water_cut_pct])

-- % Wells with Anomaly
Anomaly Well Pct =
    DIVIDE(
        DISTINCTCOUNT(anomaly_events_report[well_id]),
        [Active Wells]
    ) * 100
```

---

## Step 3: Page 1 — Field Overview

### Layout (use 16:9 canvas):

```
┌─────────────────────────────────────────────────────────┐
│  🛢️ Petroleum Production Dashboard    [Date Slicer]       │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ Total    │ Avg Rate │ Active   │ Water    │  YoY        │
│ Prod     │ bbl/day  │ Wells    │ Cut Avg  │  Change     │
│ KPI Card │ KPI Card │ KPI Card │ KPI Card │  KPI Card   │
├──────────────────────────────┬──────────────────────────┤
│                              │                          │
│  Area Chart:                 │  Stacked Bar:            │
│  Total Field Rate Over Time  │  Production by Basin     │
│  (date on X, rate on Y)      │  (basin on Y, bbl on X)  │
│                              │                          │
├──────────────────────────────┴──────────────────────────┤
│  Line Chart: Water Cut Trend by Basin (date vs wc_pct)  │
└─────────────────────────────────────────────────────────┘
```

### Visuals to add:

1. **KPI Cards** (5 cards in a row):
   - Measure: `Total Oil Production`, format as `#,0 bbls`
   - Measure: `Avg Daily Rate`, format as `#,0 bbl/day`
   - Measure: `Active Wells`
   - Measure: `Avg Water Cut`, format as `0.0%`
   - Measure: `YoY Change %` with conditional formatting (green/red)

2. **Area Chart** (bottom-left):
   - X-axis: `date` (Month hierarchy)
   - Y-axis: `Total Oil Production`
   - Legend: `basin`
   - Title: "Monthly Production by Basin"

3. **Clustered Bar** (bottom-right):
   - Y-axis: `well_id`
   - X-axis: `Total Oil Production`
   - Sort by value descending
   - Title: "Cumulative Production by Well"

4. **Slicers** (top-right):
   - `year` (dropdown)
   - `basin` (list, multi-select)

---

## Step 4: Page 2 — Well Performance & Forecast

### Visuals:

1. **Line Chart: Individual Well Decline**
   - X-axis: `production_month`
   - Y-axis: `oil_rate_bbl_day`
   - Legend: `well_id`
   - Add slicer to filter to one well at a time
   - Title: "Well Production Profile (select well)"

2. **Table: DCA Results** (from `dca_summary` if imported)
   - Columns: well_id, basin, decline_model, qi_bbl_day, Di_per_month, r2_fit, forecast_cum_12mo_bbl
   - Conditional formatting on r2_fit: Green > 0.85, Yellow 0.70–0.85, Red < 0.70

3. **Bar Chart: 12-Month Forecast Ranking**
   - Y-axis: `well_id`
   - X-axis: `forecast_cum_12mo_bbl`
   - Data labels on
   - Sort descending

4. **Gauge Visual: Field Recovery Rate**
   - Value: `Total Oil Production`
   - Max: set to 120% of historical max
   - Title: "Field Utilization"

---

## Step 5: Page 3 — Anomaly & Workover Report

### Visuals:

1. **Matrix (Heatmap): Deviation by Well × Month**
   - Rows: `well_id`
   - Columns: `date` (month)
   - Values: `deviation_pct`
   - Conditional formatting: Diverging scale — Red (−100%) → White (0%) → Green (+50%)

2. **Scatter Plot: Anomaly Events**
   - X-axis: `date`
   - Y-axis: `oil_rate_bbl_day`
   - Legend: `anomaly_type`
   - Size: `|deviation_pct|` (absolute deviation)
   - Filter to `anomaly_detected = TRUE`

3. **Table: Anomaly Events Report**
   - Source: `anomaly_events_report.csv`
   - Columns: date, well_id, basin, oil_rate_bbl_day, rolling_baseline_bbl_day, deviation_pct, anomaly_type
   - Conditional formatting on anomaly_type:
     - "Critical Drop" → Red background
     - "Moderate Drop" → Orange background
     - "Spike"         → Purple background

4. **Funnel Chart: Workover Priority**
   - Category: `workover_priority`
   - Values: count of wells

5. **Cards:**
   - `Anomaly Count`
   - `Anomaly Well Pct`
   - "Workover Candidates: 2 High Priority"

---

## Step 6: Formatting & Theme

Apply this color theme for a professional petroleum industry look:

```json
{
  "name": "Petroleum Theme",
  "dataColors": [
    "#1565C0", "#0288D1", "#00796B",
    "#558B2F", "#F57F17", "#E65100", "#B71C1C"
  ],
  "background": "#FAFAFA",
  "foreground": "#212121",
  "tableAccent": "#1565C0"
}
```

Save as `petroleum_theme.json` and load via:
**View → Themes → Browse for themes**

---

## Step 7: Publishing

1. Save as `Petroleum_Production_Dashboard.pbix`
2. Publish to Power BI Service (if available)
3. Export screenshots as PNG for portfolio/resume:
   - File → Export → Export to PDF

---

## Portfolio Presentation Tips

When describing this dashboard in interviews:

> *"I built a 3-page Power BI dashboard analyzing 12 wells across 3 Indian basins
> over 24 months. The Field Overview page uses DAX measures to show KPI cards and
> basin-level production trends. The Forecast page visualizes decline curve analysis
> results from Python, giving 12-month production forecasts per well. The Anomaly page
> uses a deviation heatmap to highlight operational issues — identifying 2 critical
> workover candidates that were 60–78% below expected rates, using Z-score and IQR
> anomaly detection methods."*
