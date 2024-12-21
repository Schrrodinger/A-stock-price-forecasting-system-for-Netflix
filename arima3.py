import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def prepare_data(df):
    """Prepare data with log transformation"""
    df_prep = df.copy()
    df_prep['log_price'] = np.log(df_prep['Close'])
    return df_prep.dropna()


def train_enhanced_model(df, train_size=0.8):
    """Train SARIMAX model"""
    # Prepare data
    df_prepared = prepare_data(df)

    # Split data
    split_idx = int(len(df_prepared) * train_size)
    train = df_prepared[:split_idx]
    test = df_prepared[split_idx:]

    # Work with log prices
    train_data = train['log_price'].dropna()
    test_data = test['log_price'].dropna()

    # Find best parameters
    best_model = auto_arima(train_data,
                            start_p=1, start_q=1,
                            max_p=3, max_q=3,
                            m=5,
                            start_P=1, start_Q=1,
                            max_P=2, max_Q=2,
                            seasonal=True,
                            d=1, D=1,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

    order = best_model.order
    seasonal_order = best_model.seasonal_order

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

    # Transform back from log space
    predictions_actual = np.exp(predictions)
    conf_int_actual = np.exp(conf_int)

    return (np.exp(train_data), np.exp(test_data),
            predictions_actual, conf_int_actual, results)


def plot_forecast(train_data, test_data, predictions, conf_int):
    """Plot only the forecasting results"""
    plt.figure(figsize=(12, 6))

    # Plot data
    plt.plot(train_data.index, train_data,
             label='Training Data', color='blue', linewidth=1)
    plt.plot(test_data.index, test_data,
             label='Actual Test Data', color='green', linewidth=1)
    plt.plot(test_data.index, predictions,
             label='Predictions', color='red', linewidth=2)

    # Plot confidence intervals
    plt.fill_between(test_data.index,
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='red', alpha=0.1)

    plt.title('Netflix Stock Price Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stock_data(df, train_size=0.8):
    """Plot the stock price data showing training and test split"""
    # Split data
    split_idx = int(len(df) * train_size)
    train = df[:split_idx]
    test = df[split_idx:]

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot training and test data
    plt.plot(train.index, train['Close'],
             label='Training Data', color='blue', linewidth=1)
    plt.plot(test.index, test['Close'],
             label='Actual Test Data', color='green', linewidth=1)

    plt.title('Netflix Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def main():
    # Load data
    df = pd.read_csv('dataset/FE_NFLX.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Train model and get predictions
    train_data, test_data, predictions, conf_int, model = train_enhanced_model(df)

    # Ensure proper data alignment
    predictions = pd.Series(predictions, index=test_data.index)
    conf_int = pd.DataFrame(conf_int, index=test_data.index)

    # Plot results
    plot_forecast(train_data, test_data, predictions, conf_int)
    plot_stock_data(df)
    return model, predictions


if __name__ == "__main__":
    model, predictions = main()