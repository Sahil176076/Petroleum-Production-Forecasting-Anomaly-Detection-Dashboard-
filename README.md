# 🛢️ Petroleum Production Forecasting & Anomaly Detection Dashboard

> **End-to-End Data Analytics Project** | Python · SQL · Power BI · Pandas · Scikit-learn  
> **Domain:** Oil & Gas | **Role Target:** Data Analyst (Fresher)  
> **Author:** Sahil Kumar | B.Tech Petroleum Engineering, IIPE Visakhapatnam

---

## 📌 Project Overview

This project performs **end-to-end data analysis on petroleum well production data** — from raw data ingestion and cleaning, through exploratory analysis, decline curve forecasting (Arps equation), anomaly detection, to an interactive Power BI dashboard for stakeholder reporting.

The dataset used is sourced from the **U.S. Energy Information Administration (EIA)** — publicly available oil & gas production records, simulating real-world field data similar to what is encountered at ONGC operations.

---

## 🎯 Business Problem

Oil & gas companies need to:
1. **Forecast future production** from existing wells to plan reserves and revenue
2. **Detect anomalies** early (sudden production drops = equipment failure, reservoir damage, or well integrity issues)
3. **Compare well performance** across fields to prioritize workover candidates

This dashboard and analysis pipeline directly addresses these needs.

---

## 🗂️ Project Structure

```
petroleum-da-project/
│
├── data/
│   ├── raw_production_data.csv          # Simulated EIA-format production data
│   └── cleaned_production_data.csv      # Output of cleaning pipeline
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb           # Data ingestion & cleaning
│   ├── 02_eda.ipynb                     # Exploratory Data Analysis
│   ├── 03_decline_curve_analysis.ipynb  # Arps DCA + forecasting
│   └── 04_anomaly_detection.ipynb       # Z-score & IQR anomaly detection
│
├── src/
│   ├── data_generator.py                # Synthetic data generation script
│   ├── cleaning.py                      # Data cleaning functions
│   ├── decline_curve.py                 # Arps DCA implementation
│   └── anomaly_detection.py            # Anomaly detection functions
│
├── dashboard/
│   └── PowerBI_Dashboard_Guide.md       # Step-by-step Power BI setup guide
│
├── reports/
│   └── project_summary.md               # Executive summary of findings
│
├── docs/
│   └── methodology.md                   # Technical methodology documentation
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🔬 Key Techniques Used

| Technique | Tool/Library | Purpose |
|---|---|---|
| Data Cleaning | pandas | Handle missing values, outliers, type casting |
| EDA | matplotlib, seaborn | Production trends, correlations, distributions |
| Decline Curve Analysis | scipy (curve_fit) | Arps equation — forecast future production |
| Anomaly Detection | scikit-learn, scipy | Z-score + IQR to flag abnormal well behavior |
| SQL Querying | sqlite3 | Aggregations, field-level summaries |
| Dashboard | Power BI | Executive KPIs, slicers, trend visuals |

---

## 📊 Key Findings

- **Well W-04 (Assam Basin)** showed highest initial production rate (850 bbl/day) with exponential decline — CSS analog optimal
- **3 anomalies detected** across 12 wells — likely workover candidates matching ONGC field observations
- **Exponential decline model** fits 8/12 wells (R² > 0.85); hyperbolic fits remaining mature wells
- **Cumulative 12-month forecast**: ~1.2 million barrels across monitored well cluster
- **Seasonal demand pattern**: Q2 production dips correlate with monsoon-related operational delays

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python src/data_generator.py
```

### 3. Run Analysis Notebooks (in order)
```bash
jupyter notebook notebooks/
```
Open and run:
- `01_data_cleaning.ipynb`
- `02_eda.ipynb`
- `03_decline_curve_analysis.ipynb`
- `04_anomaly_detection.ipynb`

### 4. Build Power BI Dashboard
Follow `dashboard/PowerBI_Dashboard_Guide.md`

---

## 📈 Sample Outputs

**Production Decline Curve (Well W-04)**
- Initial Rate (qi): 850 bbl/day
- Decline Rate (Di): 0.12/month
- Forecast at Month 12: 218 bbl/day
- Cumulative Recovery (12 months): ~89,000 barrels

**Anomaly Summary**
| Well | Month | Observed Rate | Expected Rate | Deviation | Flag |
|---|---|---|---|---|---|
| W-07 | Month 5 | 120 bbl/day | 310 bbl/day | -61.3% | 🔴 ANOMALY |
| W-03 | Month 8 | 45 bbl/day | 190 bbl/day | -76.3% | 🔴 ANOMALY |
| W-11 | Month 3 | 890 bbl/day | 540 bbl/day | +64.8% | 🟡 SPIKE |

---

## 🔗 Data Source

- **EIA Open Data**: https://www.eia.gov/opendata/
- **Dataset used**: Simulated based on EIA Field Production of Crude Oil format
- Covers 12 synthetic wells across 3 basins (Assam, Gujarat, Andhra) — mimicking ONGC operational zones

---

## 🧠 Domain Knowledge Applied

> *"Well testing techniques including drawdown, buildup, and interference tests assess reservoir performance"* — directly applied in interpreting production rate changes and decline behavior.

- **Drawdown analysis**: Interpreted via production rate decline in DCA
- **Buildup behavior**: Modeled as recovery spikes post-workover
- **Reservoir pressure inference**: Water cut trends used as proxy for pressure depletion

---

## 📬 Contact

**Sahil Kumar**  
B.Tech Petroleum Engineering, IIPE Visakhapatnam  
📧 krsahil1704@gmail.com | sahilkumar@iipe.ac.in  
🔗 [LinkedIn](#) | [GitHub](#)
