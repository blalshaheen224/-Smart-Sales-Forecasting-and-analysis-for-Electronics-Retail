import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
def load_prepare_data(path):
    df = pd.read_csv(path ,parse_dates=["Date"])
    df["Revenue"] =df['Price'] * df['Units_Sold']
    for col in ['Category' , 'Product_Name' , 'Branch']:
      df[col] = df[col].astype('category')
    return df

def plot_sales_by_category(df):
    sales_by_category = df.groupby("Category")["Units_Sold"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_by_category.index, y=sales_by_category.values)
    plt.xlabel('Category')
    plt.ylabel('Total Units Sold')
    plt.title('Total Units Sold by Category')
    plt.xticks(rotation=45)
    plt.show()

def plot_top_products(df, top_n=10):
  top_produt = df.groupby('Product_Name')['Units_Sold'].sum().sort_values(ascending=False).head(top_n)
  plt.figure(figsize=(10,6))
  sns.barplot(x = top_produt.index , y = top_produt.values )
  plt.xlabel('Product Name')
  plt.ylabel('Total Units_Sold')
  plt.title(f'Top {top_n} Products by Units_Sold')
  plt.xticks(rotation=45)
  plt.show()

def plot_sales_by_branch(df):
    branch_sales = df.groupby("Branch")["Units_Sold"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=branch_sales.index, y=branch_sales.values)
    plt.title('Top  Branch by Units_Sold')
    plt.ylabel('Total Units_Sold')
    plt.xlabel("Branch")
    plt.show()

def plot_sales_over_time(df):
  data_series = df.resample("MS", on = "Date")["Units_Sold"].sum()
  plt.figure(figsize=(12,6))
  sns.lineplot(x=data_series.index , y = data_series.values)
  plt.xlabel('Date')
  plt.ylabel('Total Units_Sold')
  plt.title('Sales Over Time')

  for x, y in zip(data_series.index, data_series.values):
    plt.axvline(x=x, ymin=0, ymax=y/max(data_series.values)-0.1, linestyle='--', color='gray', alpha=0.5)

  import matplotlib.dates as mdates
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1),)
  plt.xticks(rotation=45 , ha = "center",fontsize=10)
  plt.show()


def encode_features_standrad(df, categorical_cols):
   encode ={}
   df_ecoded = df.copy()
   for col in categorical_cols:
       encoder = LabelEncoder()
       df_ecoded[col] = encoder.fit_transform(df[col])
       encode[col] = encoder
   df_ecoded[['Price']] = StandardScaler().fit_transform(df_ecoded[['Price']])
   return df_ecoded, encode

def train_model(df):
  df['Day'] = pd.to_datetime(df['Date']).dt.day
  df['Month'] = pd.to_datetime(df['Date']).dt.month
  df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday

  categorical_cols = ['Category', 'Product_Name', 'Branch']
  df_encoded , _ = encode_features_standrad(df, categorical_cols)

  X = df_encoded.drop(['Units_Sold', 'Date','Product_ID','Revenue'], axis=1)
  y = df_encoded['Units_Sold']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  print(f"Root Mean Squared Error: {rmse}")
  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2}")
  return model ,X

def feature_importance_analysis(df,model,X):
  feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
  plt.figure(figsize=(10, 6))
  sns.barplot(x=feature_importance, y=feature_importance.index)
  plt.xlabel('Feature Importance')
  plt.ylabel('Features')
  plt.title('Feature Importance Analysis')

def predict_next_month_sales(df, model, encoders):
  future = []
  latest_data = pd.to_datetime(df['Date'].max())
  for product in df['Product_Name'].unique():
    for branch in df['Branch'].unique():
      for Days in range(1,31):
        future_date = latest_data + pd.Timedelta(days=Days)
        price = df[(df['Product_Name'] == product) & (df['Branch'] == branch)]['Price']
        future.append({
                    'Product_Name': product,
                    'Category': df[df['Product_Name'] == product]['Category'].mode()[0],
                    'Price': df[df['Product_Name'] == product]['Price'].mean(),
                    'Branch': branch,
                    'Day': future_date.day,
                    'Month': future_date.month,
                    'Weekday': future_date.weekday()
                })

  future_df = pd.DataFrame(future)
  for col, le in encoders.items():
    future_df[col] = le.transform(future_df[col])
  future_df[['Price']] = StandardScaler().fit_transform(future_df[['Price']])
  X_future = future_df[['Product_Name','Category', 'Price', 'Branch', 'Day', 'Month', 'Weekday']]
  future_df['Predicted_Units_Sold'] = model.predict(X_future)
  print("\n predict of future_data for month")
  for col ,le in encoders.items():
    future_df[col] = le.inverse_transform(future_df[col])
  preduction = future_df.groupby(['Product_Name','Branch'])['Predicted_Units_Sold'].sum().round().astype(int)
  return preduction


def run_pipeline():
    df = pd.read_csv('/content/electronics_sales_data.csv')
    df['Revenue'] = df['Units_Sold'] * df['Price']
    categorical_cols = ['Category', 'Product_Name', 'Branch']
    df_encoder, encoders = encode_features_standrad(df,categorical_cols)
    model,X = train_model(df_encoder)
    feature_importance_analysis(df,model,X)
    predictions = predict_next_month_sales(df, model, encoders)

    return predictions

def run_expler():
   df = load_prepare_data('/content/electronics_sales_data.csv')
   plot_sales_by_category(df)
   plot_top_products(df,10)
   plot_sales_by_branch(df)
   plot_sales_over_time(df)
run_expler()
run_pipeline()
