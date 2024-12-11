import pandas as pd
import os

# Read the CSV file
df = pd.read_csv('NFLX.csv')

# Check for N/A values
print("Initial N/A values:")
print(df.isna().sum())

# Interpolate any missing values
# For numeric columns, this will use linear interpolation
df_preprocessed = df.copy()
df_preprocessed = df_preprocessed.interpolate()

# Verify no more N/A values remain
print("\nN/A values after interpolation:")
print(df_preprocessed.isna().sum())

# Ensure the Date column is in datetime format
df_preprocessed['Date'] = pd.to_datetime(df_preprocessed['Date'])

# Sort by date to ensure proper interpolation
df_preprocessed = df_preprocessed.sort_values('Date')

# Save the preprocessed data to a new CSV file
output_path = 'preprocessed_data.csv'
df_preprocessed.to_csv(output_path, index=False)

print(f"\nPreprocessed data saved to {os.path.abspath(output_path)}")

# Display the first few rows of the preprocessed data
print("\nPreprocessed Data Preview:")
print(df_preprocessed.head())