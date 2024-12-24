from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, window, expr
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Enhanced Stock Price Analysis") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

def load_and_preprocess_data(file_path):
    """Load and preprocess the data with validation"""
    print("\n=== Data Loading and Preprocessing ===")
    
    # Read CSV file using Spark
    spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Keep only essential columns and ensure proper types
    spark_df = spark_df.select(
        col("Date").cast("timestamp").alias("Date"),
        col("Close").cast("double").alias("Close")
    )
    
    # Sort by date
    spark_df = spark_df.orderBy("Date")
    
    # Convert to pandas for time series processing
    pdf = spark_df.toPandas()
    pdf.set_index('Date', inplace=True)
    pdf.sort_index(inplace=True)
    
    # Handle missing values using forward fill
    pdf['Close'] = pdf['Close'].fillna(method='ffill')
    
    print(f"Dataset shape: {pdf.shape}")
    print("\nFirst few rows:")
    print(pdf.head())
    
    return pdf

def train_test_split(df, train_size=0.8):
    """Split data into train and test sets"""
    split_idx = int(len(df) * train_size)
    train = df[:split_idx].copy()
    test = df[split_idx:].copy()
    
    print(f"\nTrain set size: {len(train)}")
    print(f"Test set size: {len(test)}")
    
    return train, test

def fit_arima_model(train_data):
    """Fit ARIMA model with error handling"""
    try:
        # Use parameters suitable for daily stock data
        order = (2, 1, 2)  # (p, d, q)
        seasonal_order = (1, 1, 1, 5)  # (P, D, Q, s)
        
        print(f"\nUsing model parameters:")
        print(f"Order: {order}")
        print(f"Seasonal Order: {seasonal_order}")
        
        model = SARIMAX(train_data,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        results = model.fit(disp=False)
        return results
    
    except Exception as e:
        print(f"Error in model fitting: {str(e)}")
        raise

def make_predictions(model_results, test_data):
    """Generate predictions with error handling"""
    try:
        predictions = model_results.get_forecast(steps=len(test_data))
        forecast = predictions.predicted_mean
        conf_int = predictions.conf_int()
        
        # Ensure predictions align with test data index
        forecast.index = test_data.index
        conf_int.index = test_data.index
        
        return forecast, conf_int
    
    except Exception as e:
        print(f"Error in making predictions: {str(e)}")
        raise

def calculate_directional_metrics(actual, predicted):
    """Calculate comprehensive directional accuracy metrics"""
    try:
        # Calculate returns
        actual_returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()
        
        # Ensure index alignment
        actual_returns = actual_returns[predicted_returns.index]
        
        # Calculate directions
        actual_direction = np.sign(actual_returns)
        predicted_direction = np.sign(predicted_returns)
        
        # Basic directional accuracy
        correct_direction = (actual_direction == predicted_direction)
        direction_accuracy = np.mean(correct_direction) * 100
        
        # Separate accuracies for up and down movements
        up_mask = actual_direction > 0
        down_mask = actual_direction < 0
        
        up_accuracy = np.mean(correct_direction[up_mask]) * 100 if any(up_mask) else 0
        down_accuracy = np.mean(correct_direction[down_mask]) * 100 if any(down_mask) else 0
        
        # Confusion matrix elements
        true_positives = np.sum((actual_direction > 0) & (predicted_direction > 0))
        true_negatives = np.sum((actual_direction < 0) & (predicted_direction < 0))
        false_positives = np.sum((actual_direction < 0) & (predicted_direction > 0))
        false_negatives = np.sum((actual_direction > 0) & (predicted_direction < 0))
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        confusion_matrix = pd.DataFrame(
            [[true_positives, false_positives],
             [false_negatives, true_negatives]],
            index=['Actual Up', 'Actual Down'],
            columns=['Predicted Up', 'Predicted Down']
        )
        
        return {
            'Overall Directional Accuracy': direction_accuracy,
            'Upward Movement Accuracy': up_accuracy,
            'Downward Movement Accuracy': down_accuracy,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1_score * 100,
            'Confusion Matrix': confusion_matrix
        }
    
    except Exception as e:
        print(f"Error calculating directional metrics: {str(e)}")
        return None

def plot_directional_analysis(actual, predicted):
    """Create visualization of directional analysis"""
    try:
        # Calculate returns
        actual_returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()
        
        # Ensure index alignment
        actual_returns = actual_returns[predicted_returns.index]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns comparison
        axes[0, 0].scatter(actual_returns, predicted_returns, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].set_title('Actual vs Predicted Returns')
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        
        # Rolling accuracy
        correct_direction = np.sign(actual_returns) == np.sign(predicted_returns)
        rolling_accuracy = pd.Series(correct_direction, index=actual_returns.index) \
                           .rolling(window=20).mean()
        axes[0, 1].plot(rolling_accuracy.index, rolling_accuracy * 100)
        axes[0, 1].set_title('20-Day Rolling Directional Accuracy')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Accuracy (%)')
        
        # Returns distribution
        axes[1, 0].hist(actual_returns, bins=50, alpha=0.5, label='Actual')
        axes[1, 0].hist(predicted_returns, bins=50, alpha=0.5, label='Predicted')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Cumulative returns
        cum_actual = (1 + actual_returns).cumprod()
        cum_predicted = (1 + predicted_returns).cumprod()
        axes[1, 1].plot(cum_actual.index, cum_actual, label='Actual')
        axes[1, 1].plot(cum_predicted.index, cum_predicted, label='Predicted')
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting directional analysis: {str(e)}")

def main():
    """Main function with comprehensive error handling"""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data('dataset/FE_NFLX_new.csv')
        
        # Split data
        train_df, test_df = train_test_split(df)
        
        # Fit model
        model_results = fit_arima_model(train_df['Close'])
        
        # Make predictions
        predictions, conf_int = make_predictions(model_results, test_df)
        
        # Calculate directional metrics
        dir_metrics = calculate_directional_metrics(test_df['Close'], predictions)
        
        if dir_metrics:
            print("\nDirectional Analysis Results:")
            print("-" * 50)
            for metric, value in dir_metrics.items():
                if metric != 'Confusion Matrix':
                    print(f"{metric:.<35} {value:>10.2f}%")
            print("\nConfusion Matrix:")
            print(dir_metrics['Confusion Matrix'])
        
        # Plot directional analysis
        plot_directional_analysis(test_df['Close'], predictions)
        
        return model_results, predictions, dir_metrics
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None, None, None
    finally:
        spark.stop()

if __name__ == "__main__":
    model, predictions, metrics = main()