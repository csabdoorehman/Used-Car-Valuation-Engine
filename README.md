# Used-Car-Valuation-Engine
An end-to-end Machine Learning pipeline  that predicts pre-owned vehicle market values using XGBoost regression and tabular feature engineering.

# 🚗 Used Car Valuation Engine

An end-to-end machine learning project designed to estimate fair market values for pre-owned vehicles based on usage condition, age, engine specifications, and manufacturer brand.

## 📌 Project Overview
The **Used Car Valuation Engine** automates the pricing process for pre-owned vehicles by transforming raw tabular data into predictive price estimates. The system builds an end-to-end Scikit-Learn `Pipeline` coupled with an `XGBoostRegressor` model and exposes the trained artifact through an interactive Streamlit UI.

## 🛠️ Key Features
* **Automated Feature Engineering:** Calculates vehicle age (`car_age`) to model non-linear depreciation curves.
* **Preprocessing Pipeline:** Integrates `ColumnTransformer` for `StandardScaler` numeric scaling and `OneHotEncoder` categorical transformations.
* **Model Benchmark:** Evaluates regression algorithms (Random Forest vs. XGBoost) using RMSE, MAE, and $R^2$ metrics.
* **Pipeline Serialization:** Packages preprocessing and model steps into a single reusable `.joblib` artifact.
* **Interactive UI:** Deploys a Streamlit web app allowing users to adjust vehicle parameters via sliders and dropdowns to receive instant valuations.

## 🧰 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit
