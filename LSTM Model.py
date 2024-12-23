from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, window, expr, desc
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


class StockPriceLSTM:
    def __init__(self, spark_session, sequence_length=60):
        """
        Initialize the LSTM model for stock price prediction

        Args:
            spark_session: Active Spark session
            sequence_length: Number of time steps to look back for prediction
        """
        self.spark = spark_session
        self.sequence_length = sequence_length
        self.model = None
        self.feature_scaler = None
        self.price_scaler = None
        self.feature_columns = [
            'Returns', 'Log_Returns', 'Price_Range_Pct',
            'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
            'RSI', 'BB_Upper', 'BB_Lower',
            'K_Line', 'D_Line',
            'Volume_SMA_5', 'Volume_Ratio',
            'MACD', 'MACD_Signal',
            'Daily_Volatility', 'ATR'
        ]

    def prepare_data(self, df):
        """
        Prepare and normalize data for LSTM model training

        Args:
            df: PySpark DataFrame with engineered features
        Returns:
            X_train, y_train, X_val, y_val
        """
        # Convert to pandas for preprocessing
        pdf = df.toPandas()

        # Initialize scalers if not already done
        if self.feature_scaler is None:
            self.feature_scaler = SklearnStandardScaler()
            self.price_scaler = SklearnStandardScaler()

        # Scale features
        scaled_features = self.feature_scaler.fit_transform(pdf[self.feature_columns])
        pdf_scaled = pd.DataFrame(scaled_features, columns=self.feature_columns, index=pdf.index)

        # Scale target (Close price)
        pdf_scaled['Close'] = self.price_scaler.fit_transform(pdf[['Close']])

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(pdf_scaled) - self.sequence_length):
            seq = pdf_scaled[self.feature_columns].iloc[i:(i + self.sequence_length)].values
            target = pdf_scaled['Close'].iloc[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Split into training and validation sets (80-20 split)
        train_size = int(len(sequences) * 0.8)

        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        X_val = sequences[train_size:]
        y_val = targets[train_size:]

        return X_train, y_train, X_val, y_val

    def build_model(self):
        """
        Build improved LSTM model architecture
        """
        self.model = Sequential([
            # First LSTM layer with more units
            LSTM(units=128, return_sequences=True,
                 input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.3),

            # Second LSTM layer
            LSTM(units=64, return_sequences=True),
            Dropout(0.3),

            # Third LSTM layer
            LSTM(units=32),
            Dropout(0.3),

            # Dense layers for better feature extraction
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])

        # Use Huber loss for better handling of outliers
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='huber',
            metrics=['mae']
        )

    def train(self, df, epochs=100, batch_size=32):
        """
        Train the LSTM model with improved parameters

        Args:
            df: PySpark DataFrame with features
            epochs: Number of training epochs
            batch_size: Batch size for training
        Returns:
            Training history
        """
        # Prepare sequences
        X_train, y_train, X_val, y_val = self.prepare_data(df)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def predict(self, df):
        """
        Make predictions using the trained model

        Args:
            df: PySpark DataFrame with features
        Returns:
            numpy array of predictions in original scale
        """
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        # Convert to pandas and scale features
        pdf = df.toPandas()
        scaled_features = self.feature_scaler.transform(pdf[self.feature_columns])
        pdf_scaled = pd.DataFrame(scaled_features, columns=self.feature_columns, index=pdf.index)

        # Create sequences
        sequences = []
        for i in range(len(pdf_scaled) - self.sequence_length):
            seq = pdf_scaled[self.feature_columns].iloc[i:(i + self.sequence_length)].values
            sequences.append(seq)

        sequences = np.array(sequences)

        # Make predictions
        scaled_predictions = self.model.predict(sequences)

        # Inverse transform predictions to original scale
        predictions = self.price_scaler.inverse_transform(scaled_predictions)

        return predictions.flatten()


class LSTMModelEvaluator:
    def __init__(self, actual_values, predicted_values, dates, history=None):
        """
        Initialize the evaluator with actual and predicted values
        """
        self.actual = actual_values
        self.predicted = predicted_values
        self.dates = dates
        self.history = history
        self.metrics = {}

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        # Ensure actual and predicted are numpy arrays
        actual = np.array(self.actual).flatten()
        predicted = np.array(self.predicted).flatten()

        self.metrics['mse'] = mean_squared_error(actual, predicted)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(actual, predicted)
        self.metrics['mape'] = np.mean(np.abs((actual - predicted) / actual)) * 100
        self.metrics['r2'] = r2_score(actual, predicted)

        # Fix directional accuracy calculation
        direction_actual = np.sign(np.diff(actual))
        direction_pred = np.sign(np.diff(predicted))

        # Ensure both arrays have the same shape before comparison
        if len(direction_actual) == len(direction_pred):
            self.metrics['directional_accuracy'] = np.mean(direction_actual == direction_pred) * 100
        else:
            self.metrics['directional_accuracy'] = np.nan
            print(
                f"Warning: Cannot calculate directional accuracy due to shape mismatch: {direction_actual.shape} vs {direction_pred.shape}")

        return self.metrics

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        if not self.metrics:
            self.calculate_metrics()

        report = """
    LSTM Model Evaluation Report
    ===========================

    1. Performance Metrics
    ---------------------
    Root Mean Square Error (RMSE): ${:.2f}
    Mean Absolute Error (MAE): ${:.2f}
    Mean Absolute Percentage Error (MAPE): {:.2f}%
    R-squared Score: {:.4f}
    Directional Accuracy: {:.2f}%

    2. Model Characteristics
    -----------------------
    - Number of predictions: {}
    - Price range: ${:.2f} - ${:.2f}
    - Prediction range: ${:.2f} - ${:.2f}

    3. Error Analysis
    ----------------
    - Standard deviation of errors: ${:.2f}
    - Mean error: ${:.2f}
    - Median error: ${:.2f}
    """.format(
            self.metrics['rmse'],
            self.metrics['mae'],
            self.metrics['mape'],
            self.metrics['r2'],
            self.metrics['directional_accuracy'],
            len(self.actual),
            np.min(self.actual),
            np.max(self.actual),
            np.min(self.predicted),
            np.max(self.predicted),
            np.std(self.actual - self.predicted),
            np.mean(self.actual - self.predicted),
            np.median(self.actual - self.predicted)
        )

        return report


    def plot_price_comparison(self, save_path="results"):
        """Plot actual vs predicted prices and save"""
        plt.figure(figsize=(15, 8))
        plt.plot(self.dates, self.actual, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(self.dates, self.predicted, label='Predicted Price', color='red', alpha=0.7)
        plt.title('Netflix Stock Price: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/price_comparison.png")
        plt.close()

    def plot_training_history(self, save_path="results"):
        """Plot training and validation loss and save"""
        if self.history is None:
            print("No training history available")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Training Loss', color='blue')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png")
        plt.close()

    def plot_error_distribution(self, save_path="results"):
        """Plot distribution of prediction errors and save"""
        errors = self.actual - self.predicted

        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/error_distribution.png")
        plt.close()

    def plot_scatter_actual_vs_predicted(self, save_path="results"):
        """Create scatter plot of actual vs predicted values and save"""
        plt.figure(figsize=(10, 10))
        plt.scatter(self.actual, self.predicted, alpha=0.5)
        plt.plot([self.actual.min(), self.actual.max()],
                 [self.actual.min(), self.actual.max()],
                 'r--', lw=2)
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/scatter_plot.png")
        plt.close()

    def plot_residuals(self, save_path="results"):
        """Plot residuals over time and save"""
        residuals = self.actual - self.predicted

        plt.figure(figsize=(15, 8))
        plt.scatter(self.dates, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot Over Time')
        plt.xlabel('Date')
        plt.ylabel('Residual ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/residuals.png")
        plt.close()

    def save_evaluation_report(self, save_path="results"):
        """Save evaluation report to a text file"""
        report = self.generate_evaluation_report()

        # Create timestamp for the report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to file
        with open(f"{save_path}/evaluation_report_{timestamp}.txt", "w") as f:
            f.write(report)

        return report


def main():
    import os

    # Create results directory if it doesn't exist
    results_dir = "LSTM model results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Netflix Stock LSTM") \
        .getOrCreate()

    # Load your trained LSTM model and predictions
    lstm_model = StockPriceLSTM(spark)

    # Read the feature-engineered data
    df = spark.read.csv("dataset/FE_NFLX.csv", header=True, inferSchema=True)

    # Train the model
    history = lstm_model.train(df)

    # Generate predictions
    predictions = lstm_model.predict(df)

    # Get actual values and dates ensuring alignment
    actual_values = df.select('Close').toPandas().iloc[60:].values.flatten()
    dates = df.select('Date').toPandas().iloc[60:].values

    # Ensure predictions and actual values have the same length
    min_len = min(len(actual_values), len(predictions))
    actual_values = actual_values[:min_len]
    predictions = predictions[:min_len]
    dates = dates[:min_len]

    # Create evaluator instance
    evaluator = LSTMModelEvaluator(actual_values, predictions, dates, history)

    # Save evaluation report
    evaluator.save_evaluation_report(results_dir)

    # Generate and save visualizations
    evaluator.plot_price_comparison(results_dir)
    evaluator.plot_training_history(results_dir)
    evaluator.plot_error_distribution(results_dir)
    evaluator.plot_scatter_actual_vs_predicted(results_dir)
    evaluator.plot_residuals(results_dir)

    print(f"Results have been saved to the '{results_dir}' directory")


if __name__ == "__main__":
    main()