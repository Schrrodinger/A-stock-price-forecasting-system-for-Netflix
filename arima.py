from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, stddev, avg, expr
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Initialize Spark Session first
spark = SparkSession.builder \
    .appName("Stock Price Forecasting") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

def prepare_features(spark_df):
    """Prepare additional features using Spark DataFrame operations"""
    # Define window specs
    window_spec = Window.orderBy("Date")
    window_20 = Window.orderBy("Date").rowsBetween(-19, 0)
    
    # Calculate features
    df_features = spark_df \
        .withColumn("Returns", 
                   (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec)) \
        .withColumn("Volatility", 
                   stddev("Returns").over(window_20)) \
        .withColumn("MA20", 
                   avg("Close").over(window_20)) \
        .withColumn("Volume_MA20", 
                   avg("Volume").over(window_20)) \
        .withColumn("Volume_Ratio", 
                   col("Volume") / col("Volume_MA20")) \
        .withColumn("Price_Momentum", 
                   col("Close") - lag("Close", 20).over(window_spec))
    
    # Drop rows with null values
    df_features = df_features.na.drop()
    
    return df_features

def detect_outliers(series, threshold=3):
    """Detect and handle outliers using Z-score method"""
    z_scores = stats.zscore(series)
    outliers = np.abs(z_scores) > threshold
    series_clean = series.copy()
    series_clean[outliers] = series.mean()
    return series_clean

def train_test_split(spark_df, train_ratio=0.8):
    """Split data into training and testing sets"""
    train_size = int(spark_df.count() * train_ratio)
    
    # Convert to pandas for time series modeling
    pdf = spark_df.toPandas()
    train = pdf.iloc[:train_size]
    test = pdf.iloc[train_size:]
    
    return train, test

def find_best_parameters(train_data):
    """Find optimal parameters for SARIMAX"""
    # Using fixed parameters since we removed auto_arima dependency
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 5)
    return order, seasonal_order

def train_enhanced_model(spark_df, train_size=0.8):
    """Train enhanced SARIMAX model"""
    # Prepare features
    df_prepared = prepare_features(spark_df)
    
    # Split data
    train, test = train_test_split(df_prepared, train_size)
    
    # Prepare target variable
    train_data = detect_outliers(train['Close'])
    test_data = test['Close']
    
    # Find best parameters
    print("Finding optimal parameters...")
    order, seasonal_order = find_best_parameters(train_data)
    
    # Train final model
    print(f"Training SARIMAX model with order {order} and seasonal_order {seasonal_order}")
    final_model = SARIMAX(train_data,
                         order=order,
                         seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    
    results = final_model.fit(disp=False)
    
    # Make predictions
    predictions = results.get_forecast(steps=len(test))
    forecast = predictions.predicted_mean
    conf_int = predictions.conf_int()
    
    return train_data, test_data, forecast, conf_int, results

def plot_enhanced_results(train, test, predictions, conf_int):
    """Plot results with confidence intervals"""
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(range(len(train)), train, label='Training Data', color='blue')
    
    # Plot test data
    plt.plot(range(len(train), len(train) + len(test)), test,
             label='Actual Test Data', color='green')
    
    # Plot predictions
    plt.plot(range(len(train), len(train) + len(test)), predictions,
             label='Predictions', color='red')
    
    # Plot confidence intervals
    plt.fill_between(range(len(train), len(train) + len(test)),
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color='gray', alpha=0.2,
                    label='95% Confidence Interval')
    
    plt.title('Enhanced Stock Price Forecasting using SARIMAX')
    plt.xlabel('Time Period')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_enhanced_model(test, predictions):
    """Calculate enhanced evaluation metrics"""
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test, predictions) * 100
    
    print("\nEnhanced Model Performance Metrics:")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    return rmse, mape

def main():
    try:
        # Read data
        spark_df = spark.read.csv('dataset/FE_NFLX_new.csv', 
                                header=True, 
                                inferSchema=True)
        
        # Convert Date column and set as index
        spark_df = spark_df.withColumn("Date", col("Date").cast("timestamp"))
        
        # Train enhanced model
        train, test, predictions, conf_int, model = train_enhanced_model(spark_df)
        
        # Evaluate model
        rmse, mape = evaluate_enhanced_model(test, predictions)
        
        # Plot results
        plot_enhanced_results(train, test, predictions, conf_int)
        
        return model, predictions
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    model, predictions = main()