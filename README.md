# VW & Audi Used Car Price Prediction Dashboard

An interactive machine learning dashboard for exploring and predicting used car prices from AutoTrader UK listings. Built with Streamlit, scikit-learn and Matplotlib.

**Live demo:** [https://car-price-estimator-arxhi92blxwccefy4xzty9.streamlit.app/]

---

## Overview

This project analyses ~24,000 Volkswagen and Audi used car listings scraped from AutoTrader UK. It walks through the full data science pipeline, from raw data exploration to a deployed, interactive price estimator which compares a baseline Linear Regression model against a Random Forest model.

---

## Key Findings

### What drives used car prices?

| Feature | Insight |
|---|---|
| 'car age' | Strongest single predictor. Price drops sharply after year 3. |
| 'mileage_per_year' | More predictive than raw mileage that normalises for age. |
| 'engineSize' | Clear premium for larger engines, especially in Audi. |
| 'brand' | Audi commands ~₤2,000 premium over VW on average. |
| 'transmission' | Automatics price higher, particularly in diesel variants. |
| 'fuelType' | Petrol dominates volume; diesel holds value better at higher mileages. |

### Model Comparison

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression (baseline) | 0.79 | ₤2,100 | ₤2,890 |
| Random Forest | 0.92 | ₤950 | ₤1,340 |

**Random Forest wins** - 55% lower MAE, explaining 92% of price variance. The improvement comes from its ability to capture non-linear interactions (e.g. mileage matters far more on older cars) that Linear Regression can't model.

### Brand breakdown

- **Audi** median price: ~₤13,500 • wider spread due to S/RS model variants
- **Volkswagen** median price: ₤11,200 • tighter distribution, more volume at lower price points
- Both brands show strong right skew - a long tail of high-value low-mileage cars

---

## Dashboard Pages

- **Overview** - KPI metrics, price distribution by brand, year vs price scatter
- **EDA** - Mileage & engine size distributions, fuel type breakdown, correlation heatmap, brand boxplots
- **Models** - Actual vs predicted charts, feature importance for both LR and RF, side-by-side metric comparison
- **Predict** - Live price estimator: configure brand, transmission, fuel type, engine size, year, and mileage to get an instant RF and LR price estimate

---

## Run Locally

'''bash
git clone https://github.com/Aayush-Bhuyan/Car-Price-Estimator.git
cd Car-Price-Estimator

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
'''

No CSV downloads needed - data loads automatically from the source repository.

---

## Project Structure

car-price-estimator/
├── app.py              
├── requirements.txt    
├── .gitignore
└── README.md

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12.6 | Core Language |
| Streamlit | Dashboard Framework |
| pandas / Numpy | Data wrangling & feature engineering |
| scikit-learn | Linear Regression, Random Forest, train/test split | 
| Matplotlib / Seaborn | All visualisations |

---

## Data Source

AutoTrader UK scrape - Audi & Volkswagen listings.
Dataset sourced by [Aayush Pratap Bhuyan](https://www.linkedin.com/in/aayush-pratap-bhuyan-9636301b4/) from [Kaggle](https://www.kaggle.com/datasets/guanhaopeng/uk-used-car-market) via [GitHub](https://github.com/Aayush-Bhuyan/Used-Car-Price-Prediction).

---

## Feature Engineering

Two derived features were created beyond the raw columns:
 - **'car_age'** - '2020 - year', clamped to a minimum of 1 to avoid division by zero
 - **'mileage_per_year'** - 'mileage / car_age', normalises usage relative to the car's age

 Categorical columns ('transmission', 'fuelType', 'brand') were one-hot encoded with 'drop_first=True'. Rows with zero engine size or prices outside ₤500-₤100,000 were dropped as outliers.