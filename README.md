# 📊 Electronics Sales Forecasting and Analysis

This project analyzes and forecasts sales for an electronics store using **dummy data**. It performs exploratory data analysis (EDA), identifies key factors affecting sales, and builds a machine learning model (XGBoost) to predict future sales for each product and branch.

---

## 🧰 Technologies Used

- Python
- pandas, matplotlib, seaborn
- scikit-learn
- XGBoost (regression model)
- LabelEncoder, StandardScaler
- Time series resampling and visualization

---

## 📁 Project Overview

### 1. `load_prepare_data(path)`
- Loads CSV data and parses the `Date` column.
- Adds a `Revenue` column (`Price * Units_Sold`).
- Converts categorical columns (`Category`, `Product_Name`, `Branch`) to categorical data types.

---

### 2. EDA Functions

#### `plot_sales_by_category(df)`
- Visualizes total units sold per product category using a bar chart.

#### `plot_top_products(df, top_n=10)`
- Displays the top-selling products by total units sold.

#### `plot_sales_by_branch(df)`
- Shows which branches sold the most units.

#### `plot_sales_over_time(df)`
- Plots monthly total units sold over time using line chart.
- Adds vertical dashed lines for better date segmentation.

---

### 3. Encoding & Feature Scaling

#### `encode_features_standrad(df, categorical_cols)`
- Applies Label Encoding on categorical columns.
- Standardizes the `Price` column using `StandardScaler`.

---

### 4. Model Training

#### `train_model(df)`
- Extracts day, month, and weekday from the `Date` column.
- Encodes categorical columns and splits the dataset into training/testing sets.
- Trains an `XGBRegressor` model.
- Evaluates using RMSE and R² score.
- Returns trained model and features used.

---

### 5. Feature Importance

#### `feature_importance_analysis(df, model, X)`
- Displays feature importances from the trained XGBoost model using a horizontal bar chart.

---

### 6. Future Sales Prediction

#### `predict_next_month_sales(df, model, encoders)`
- Generates a future dataset for the next month (for each day/product/branch).
- Applies same encoding/scaling.
- Predicts expected units sold using the model.
- Groups and sums predictions by `Product_Name` and `Branch`.

---

### 7. Pipeline Functions

#### `run_expler()`
- Runs the EDA functions to explore data visually.

#### `run_pipeline()`
- Loads and processes the dataset.
- Trains the model and outputs predictions for next month.
- Performs feature importance analysis.

---

## 📈 Example Use Case

With this pipeline, you can:
- Understand your best-selling products, branches, and categories.
- Discover sales patterns over time.
- Forecast next month’s demand for each product in each branch.
- Make data-driven purchase and inventory decisions.

---

## 📂 Dataset Note

This project uses **dummy sales data** for demonstration purposes. Replace it with your real store data (following the same structure) to make it production-ready.

---

## 🧪 How to Run

```bash
# Install required libraries
pip install pandas matplotlib seaborn scikit-learn xgboost

# Then run the script
python your_script.py
```

---

## 📌 Future Improvements

- Add seasonality decomposition using Prophet or statsmodels.
- Visualize data using a heatmap (e.g., sales per weekday/month).
- Deploy the model with a web interface (e.g., using Streamlit).

---

## 👤 Author

Developed by blal shaheen.  
This is a personal project to practice machine learning on retail data.
