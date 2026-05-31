# Geopolitical Credit Risk Early Warning System

ML-based early warning tool quantifying geopolitical risk transmission
to corporate PD for Irish unlisted SMEs.

## Project structure

```
geopolitical_credit_risk/
├── data/
│   ├── raw/            # Downloaded source data (gitignored)
│   ├── processed/      # Cleaned, merged datasets (gitignored)
│   └── external/       # Trade weights, sector mappings
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── data/           # Data loaders (CRO, GPR, Eurostat, GDELT)
│   ├── features/       # Feature engineering incl. sGPR index
│   ├── models/         # Training pipeline, evaluation, MLflow
│   └── visualization/  # SHAP plots, EDA charts
├── app/
│   └── app.py          # Streamlit EWS dashboard
├── outputs/            # Saved models, charts (gitignored)
└── tests/
```

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd geopolitical_credit_risk

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install project as editable package
pip install -e .

# 5. Copy env file
cp .env.example .env
```

## Running

```bash
# Notebooks (EDA, modelling)
jupyter notebook

# Training pipeline
python -m src.models.train

# MLflow UI (view experiment results)
mlflow ui

# Streamlit app
streamlit run app/app.py
```

## Data sources

| Source | Data | Access |
|--------|------|--------|
| CRO Ireland | Company financial statements | opendata.cro.ie (CC BY 4.0) |
| Caldara & Iacoviello | GPR index (monthly) | matteoiacoviello.com (free) |
| Eurostat | Insolvency, GDP, trade | ec.europa.eu/eurostat (free API) |
| GDELT | News sentiment signals | gdeltproject.org (free) |
| CSO Ireland | Trade exposure weights | cso.ie (free) |

## Sprint roadmap

- [x] Sprint 1 — Environment setup & data acquisition
- [ ] Sprint 2 — EDA & data cleaning
- [ ] Sprint 3 — Feature engineering (sGPR index)
- [ ] Sprint 4 — Model training & comparison
- [ ] Sprint 5 — SHAP explainability
- [ ] Sprint 6 — Streamlit EWS app
