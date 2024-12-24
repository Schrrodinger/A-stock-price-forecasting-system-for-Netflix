from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Netflix Stock Analysis") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

def prepare_data(spark_df):
    """Prepare data with log transformation using Spark"""
    # Add log transformation
    df_prep = spark_df.withColumn("log_price", log(col("Close")))
    
    # Convert to pandas for time series processing
    pdf = df_prep.toPandas()
    pdf['Date'] = pd.to_datetime(pdf['Date'])
    pdf.set_index('Date', inplace=True)
    pdf.sort_index(inplace=True)
    
    return pdf.dropna()

def train_enhanced_model(spark_df, train_size=0.8):
    """Train SARIMAX model"""
    try:
        # Prepare data
        df_prepared = prepare_data(spark_df)
        
        # Split data
        split_idx = int(len(df_prepared) * train_size)
        train = df_prepared[:split_idx]
        test = df_prepared[split_idx:]
        
        # Work with log prices
        train_data = train['log_price']
        test_data = test['log_price']
        
        # Use fixed parameters suitable for stock data
        order = (2, 1, 2)      # (p, d, q)
        seasonal_order = (1, 1, 1, 5)  # (P, D, Q, s)
        
        print(f"\nUsing model parameters:")
        print(f"ARIMA Order: {order}")
        print(f"Seasonal Order: {seasonal_order}")
        
        # Train model
        model = SARIMAX(train_data,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        results = model.fit(disp=False)
        
        # Generate forecasts
        forecast = results.get_forecast(steps=len(test_data))
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Ensure predictions and confidence intervals have the correct index
        predictions.index = test_data.index
        conf_int.index = test_data.index
        
        # Transform back from log space
        train_actual = np.exp(train_data)
        test_actual = np.exp(test_data)
        predictions_actual = np.exp(predictions)
        conf_int_actual = np.exp(conf_int)
        
        return train_actual, test_actual, predictions_actual, conf_int_actual, results
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def plot_forecast(train_data, test_data, predictions, conf_int):
    """Plot forecasting results"""
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    plt.plot(train_data.index, train_data.values,
             label='Training Data', color='blue', linewidth=1)
    
    # Plot test data
    plt.plot(test_data.index, test_data.values,
             label='Actual Test Data', color='green', linewidth=1)
    
    # Plot predictions
    plt.plot(predictions.index, predictions.values,
             label='Predictions', color='red', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(conf_int.index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='red', alpha=0.1,
                    label='95% Confidence Interval')
    
    plt.title('Netflix Stock Price Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print some validation info
    print(f"\nPlot ranges:")
    print(f"Training data: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Test data: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Predictions: {predictions.index.min()} to {predictions.index.max()}")
    
    plt.show()

def main():
    try:
        # Load data
        spark_df = spark.read.csv('dataset/FE_NFLX_new.csv', 
                                header=True, 
                                inferSchema=True)
        
        # Convert Date column
        spark_df = spark_df.withColumn("Date", 
                                     col("Date").cast("timestamp"))
        
        # Train model and get predictions
        train_data, test_data, predictions, conf_int, model = \
            train_enhanced_model(spark_df)
        
        # Plot results with debugging information
        print("\nData shapes:")
        print(f"Train data: {len(train_data)}")
        print(f"Test data: {len(test_data)}")
        print(f"Predictions: {len(predictions)}")
        
        plot_forecast(train_data, test_data, predictions, conf_int)
        
        return model, predictions
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    model, predictions = main()