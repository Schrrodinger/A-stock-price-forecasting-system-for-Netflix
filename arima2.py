import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path):
    """Load and preprocess the data with validation"""
    print("\n=== Data Loading and Preprocessing ===")
    df = pd.read_csv(file_path)

    # Keep only essential columns initially to reduce complexity
    essential_cols = ['Date', 'Close']
    df = df[essential_cols]

    # Convert to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Sort index to ensure temporal ordering
    df = df.sort_index()

    # Check for and handle any missing values
    if df['Close'].isnull().any():
        print("Warning: Found missing values in Close prices")
        df['Close'] = df['Close'].interpolate(method='linear')

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    return df


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
        # Find best parameters
        print("\nFinding optimal parameters...")
        best_model = auto_arima(train_data,
                                start_p=0, start_q=0,
                                max_p=3, max_q=3,
                                m=5,
                                seasonal=True,
                                d=1, D=1,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True,
                                max_order=None)

        order = best_model.order
        seasonal_order = best_model.seasonal_order

        print(f"\nBest model parameters:")
        print(f"Order: {order}")
        print(f"Seasonal Order: {seasonal_order}")

        # Fit the model
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


def make_predictions(model_results, test_length):
    """Generate predictions with error handling"""
    try:
        forecast = model_results.get_forecast(steps=test_length)
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Verify predictions are valid
        if predictions.isnull().any():
            raise ValueError("NaN values in predictions")

        return predictions, conf_int

    except Exception as e:
        print(f"Error in making predictions: {str(e)}")
        raise


def calculate_directional_metrics(actual, predicted):
    """
    Calculate comprehensive directional accuracy metrics

    Parameters:
    actual (pd.Series): Actual stock prices
    predicted (pd.Series): Predicted stock prices

    Returns:
    dict: Dictionary containing directional accuracy metrics
    """
    try:
        # Ensure inputs are pandas Series with datetime index
        actual = pd.Series(actual) if not isinstance(actual, pd.Series) else actual
        predicted = pd.Series(predicted) if not isinstance(predicted, pd.Series) else predicted

        # Calculate actual and predicted returns
        actual_returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()

        # Align the series
        actual_returns, predicted_returns = actual_returns.align(predicted_returns)

        # Remove any remaining NaN values
        mask = ~(actual_returns.isna() | predicted_returns.isna())
        actual_returns = actual_returns[mask]
        predicted_returns = predicted_returns[mask]

        if len(actual_returns) == 0:
            raise ValueError("No valid data points after cleaning")

        # Calculate directional movements
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

        # Calculate confusion matrix elements
        true_positives = np.sum((actual_direction > 0) & (predicted_direction > 0))
        true_negatives = np.sum((actual_direction < 0) & (predicted_direction < 0))
        false_positives = np.sum((actual_direction < 0) & (predicted_direction > 0))
        false_negatives = np.sum((actual_direction > 0) & (predicted_direction < 0))

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Create confusion matrix as DataFrame
        confusion_matrix = pd.DataFrame(
            [[true_positives, false_positives],
             [false_negatives, true_negatives]],
            index=['Actual Up', 'Actual Down'],
            columns=['Predicted Up', 'Predicted Down']
        )

        # Calculate magnitude-weighted accuracy
        magnitude_accuracy = np.mean(
            correct_direction * np.abs(actual_returns)
        ) / np.mean(np.abs(actual_returns)) * 100

        metrics = {
            'Overall Directional Accuracy': direction_accuracy,
            'Upward Movement Accuracy': up_accuracy,
            'Downward Movement Accuracy': down_accuracy,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1_score * 100,
            'Magnitude-Weighted Accuracy': magnitude_accuracy,
            'Confusion Matrix': confusion_matrix
        }

        return metrics

    except Exception as e:
        print(f"Error calculating directional metrics: {str(e)}")
        return None


def plot_directional_analysis(actual, predicted):
    """
    Create visualization of directional accuracy analysis

    Parameters:
    actual (pd.Series): Actual stock prices
    predicted (pd.Series): Predicted stock prices
    """
    try:
        # Calculate returns
        actual_returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()

        # Align series
        actual_returns, predicted_returns = actual_returns.align(predicted_returns)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Returns comparison
        axes[0, 0].scatter(actual_returns, predicted_returns, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].set_title('Actual vs Predicted Returns')
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')

        # 2. Directional accuracy over time
        correct_direction = np.sign(actual_returns) == np.sign(predicted_returns)
        rolling_accuracy = pd.Series(correct_direction).rolling(window=20).mean()
        axes[0, 1].plot(rolling_accuracy.index, rolling_accuracy * 100)
        axes[0, 1].set_title('20-Day Rolling Directional Accuracy')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Accuracy (%)')

        # 3. Returns distribution
        axes[1, 0].hist(actual_returns, bins=50, alpha=0.5, label='Actual')
        axes[1, 0].hist(predicted_returns, bins=50, alpha=0.5, label='Predicted')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # 4. Cumulative returns
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
        df = load_and_preprocess_data('dataset/FE_NFLX.csv')

        # Split data
        train_df, test_df = train_test_split(df)

        # Fit model
        model_results = fit_arima_model(train_df['Close'])

        # Make predictions
        predictions, conf_int = make_predictions(model_results, len(test_df))

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


if __name__ == "__main__":
    model, predictions, metrics = main()