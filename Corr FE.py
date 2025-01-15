import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def analyze_correlations(csv_path):
    """
    Perform comprehensive correlation analysis on engineered features.

    Parameters:
    csv_path (str): Path to the CSV file with engineered features
    """
    # Read the dataset
    df = pd.read_csv(csv_path)

    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Drop any non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr(method='pearson')

    # Create correlation heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5})

    plt.title('Correlation Heatmap of Technical Indicators', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Identify highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:  # threshold for high correlation
                high_corr_pairs.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })

    # Convert to DataFrame for better visualization
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)

    # Calculate correlations with Returns (target variable)
    if 'Returns' in correlation_matrix.columns:
        returns_corr = correlation_matrix['Returns'].sort_values(key=abs, ascending=False)
    else:
        returns_corr = pd.Series(dtype=float)
        print("Warning: 'Returns' column not found in dataset")

    # Analyze feature groups
    feature_groups = {
        'Price_Based': ['Close', 'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 'BB_Middle', 'BB_Upper', 'BB_Lower'],
        'Volume_Based': ['Volume', 'Volume_SMA_5', 'Volume_SMA_20', 'Volume_Ratio'],
        'Momentum': ['RSI', 'MACD', 'MACD_Signal', 'K_Line', 'D_Line'],
        'Volatility': ['Daily_Volatility', 'ATR', 'Price_Range', 'Price_Range_Pct']
    }

    group_correlations = {}
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in numeric_df.columns]
        if available_features:
            group_corr = numeric_df[available_features].corr()
            group_correlations[group_name] = group_corr

    return {
        'correlation_matrix': correlation_matrix,
        'high_correlations': high_corr_df,
        'returns_correlations': returns_corr,
        'group_correlations': group_correlations
    }


def print_correlation_analysis(analysis_results):
    """
    Print detailed correlation analysis results
    """
    print("\n=== Correlation Analysis Results ===")

    print("\n1. Top Features Correlated with Returns:")
    if not analysis_results['returns_correlations'].empty:
        print(analysis_results['returns_correlations'].head(10))

    print("\n2. Highly Correlated Feature Pairs (|correlation| > 0.8):")
    if not analysis_results['high_correlations'].empty:
        print(analysis_results['high_correlations'])
    else:
        print("No feature pairs with correlation > 0.8 found")

    print("\n3. Feature Group Analysis:")
    for group_name, group_corr in analysis_results['group_correlations'].items():
        print(f"\n{group_name} Features Average Correlation:")
        print(group_corr.mean().mean())


def main():
    # Specify your CSV file path
    csv_path = "dataset/FE_NFLX_new.csv"

    # Perform correlation analysis
    results = analyze_correlations(csv_path)

    # Print results
    print_correlation_analysis(results)

    print("\nCorrelation heatmap has been saved as 'correlation_heatmap.png'")


if __name__ == "__main__":
    main()