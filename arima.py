import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def prepare_features(df):
    """Prepare additional features for the model"""
    feature_df = df.copy()

    # Technical indicators
    feature_df['Returns'] = feature_df['Close'].pct_change()
    feature_df['Volatility'] = feature_df['Returns'].rolling(window=20).std()
    feature_df['MA20'] = feature_df['Close'].rolling(window=20).mean()

    # Volume features
    feature_df['Volume_MA20'] = feature_df['Volume'].rolling(window=20).mean()
    feature_df['Volume_Ratio'] = feature_df['Volume'] / feature_df['Volume_MA20']

    # Price momentum
    feature_df['Price_Momentum'] = feature_df['Close'] - feature_df['Close'].shift(20)

    # Remove NaN values
    feature_df = feature_df.dropna()
    return feature_df


def detect_outliers(series, threshold=3):
    """Detect and handle outliers using Z-score method"""
    z_scores = stats.zscore(series)
    outliers = np.abs(z_scores) > threshold
    series_clean = series.copy()
    series_clean[outliers] = series.mean()
    return series_clean


def find_best_parameters(train_data):
    """Find optimal parameters using auto_arima"""
    model = auto_arima(train_data,
                       start_p=0, start_q=0,
                       max_p=3, max_q=3,
                       m=5,  # seasonal period
                       start_P=0, start_Q=0,
                       max_P=2, max_Q=2,
                       seasonal=True,
                       d=1, D=1,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    return model


def train_enhanced_model(df, train_size=0.8):
    """Train enhanced SARIMAX model"""
    # Prepare features
    df_prepared = prepare_features(df)

    # Create train/test split
    train_size = int(len(df_prepared) * train_size)
    train = df_prepared[:train_size]
    test = df_prepared[train_size:]

    # Prepare target variable (Close price)
    train_data = detect_outliers(train['Close'])
    test_data = test['Close']

    # Find best parameters
    print("Finding optimal parameters...")
    best_model = find_best_parameters(train_data)

    # Get the order and seasonal order from auto_arima
    order = best_model.order
    seasonal_order = best_model.seasonal_order

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


def main(df):
    # Train enhanced model
    train, test, predictions, conf_int, model = train_enhanced_model(df)

    # Evaluate model
    rmse, mape = evaluate_enhanced_model(test, predictions)

    # Plot results
    plot_enhanced_results(train, test, predictions, conf_int)

    return model, predictions


# Example usage
if __name__ == "__main__":
    # Read your data
    df = pd.read_csv('dataset/FE_NFLX.csv')  # Replace with your data loading
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Run the enhanced analysis
    model, predictions = main(df)