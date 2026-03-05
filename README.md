# Bankruptcy Prediction — Case Analysis

## Purpose

This project analyses financial data from Polish companies to predict the probability of bankruptcy using machine learning. The dataset originates from the Emerging Markets Information Service (EMIS) and covers bankrupt companies from 2000–2012 and operating companies from 2007–2013.

## Dataset

- **File:** `bankruptcy_Train.csv`
- **Records:** 10 000 companies (≈ 2 % bankrupt, 98 % survived)
- **Features:** 64 financial ratios (renamed in the notebook for readability) covering sales, profitability, working capital, and capital structure

## Notebook Workflow (`Case_analysis.ipynb`)

1. **Data loading & cleaning** — imports the CSV, renames attributes, replaces infinite values with NaN.
2. **Descriptive statistics** — summary stats, missing-value counts, and class distribution (pie chart).
3. **Feature grouping & standardisation** — groups ratios into four categories (Sales, Profitability, Working Capital, Capital Structure) and computes z-score–normalised group scores.
4. **Correlation analysis** — heatmap of inter-group correlations.
5. **Winsorisation** — clips extreme values at the 2.5 % and 97.5 % percentiles.
6. **Distribution analysis** — histograms comparing bankrupt vs. survived firms on key ratios.
7. **Quartile-binned bankruptcy rates** — crosstabs showing bankruptcy percentage within quartiles of working capital, costs/sales, long-term debt/equity, and inventory turnover.
8. **Standardised Mean Difference (SMD)** — ranks all attributes by effect size between bankrupt and non-bankrupt groups; selects the top 15 for modelling.
9. **XGBoost model** — trains an XGBClassifier on all numeric features, evaluates across multiple probability thresholds, and displays confusion matrices.
10. **Logistic Regression model** — trains a balanced logistic regression on the top-15 SMD features, compares threshold-based performance.

## Key Libraries

- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (StandardScaler, LogisticRegression, Pipeline, GridSearchCV, metrics)
- XGBoost

## How to Run

1. Place `bankruptcy_Train.csv` in the same directory as the notebook.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy`
3. Open `Case_analysis.ipynb` and run all cells sequentially.
