import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Engineer technical features from stock price data.

    Parameters:
    df (pd.DataFrame): DataFrame with columns: Date, Open, High, Low, Close, Volume

    Returns:
    pd.DataFrame: DataFrame with additional technical indicators
    """
    # Create copy of dataframe
    df = df.copy()

    # Basic Price Features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']).diff()
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = df['Price_Range'] / df['Close']

    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['K_Line'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['D_Line'] = df['K_Line'].rolling(window=3).mean()

    # Volume Features
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']

    # Momentum Indicators
    df['MACD'] = df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26,
                                                                                                adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volatility Features
    df['Daily_Volatility'] = df['Returns'].rolling(window=20).std()
    df['ATR'] = calculate_atr(df)

    df = df.fillna(0)

    return df


def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(window=period).mean()

def main():

    print("Starting feature engineering...")
    df = pd.read_csv("preprocessed_exp.csv")
    print("Data loaded, shape:", df.shape)

    # Engineer features
    df_features = engineer_features(df)
    print("Features engineered, new shape:", df_features.shape)

    # Print final info about missing values
    print("\nMissing values after feature engineering:")
    print(df_features.isnull().sum())

    # Save the new dataset
    df_features.to_csv("FE_NFLX.csv")
    print("Data saved to FE_NFLX.csv")

    return df_features

if __name__ == "__main__":
    main()