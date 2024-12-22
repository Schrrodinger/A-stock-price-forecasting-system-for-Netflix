import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib

# Load the data
df = pd.read_csv('dataset/FE_NFLX.csv', parse_dates=['Date'])

# Data Preprocessing
# Drop rows with missing values
df.dropna(inplace=True)

# Feature Selection and Preparation
# Let's predict 'Close' price using other numerical features
features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Log_Returns',
            'Price_Range', 'Price_Range_Pct', 'SMA_5', 'SMA_20',
            'EMA_5', 'EMA_20', 'RSI']

# Remove features with all zeros
features = [f for f in features if df[f].sum() != 0]

# Prepare X (features) and y (target)
X = df[features]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model with metadata
model_metadata = {
    'model': model,
    'features': features,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'performance_metrics': {
        'MAPE': mape,
        'R2': r2
    }
}

# Save the model and metadata
joblib.dump(model_metadata, 'linear_regression_model.pkl')

# Print model performance metrics
print(f"Best model: Linear Regression")
print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
print(f"R-squared (R2) Score: {r2:.4f}")

# OLS Summary Table
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print("\nOLS Summary:")
print(model_sm.summary())

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualization of Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Linear Regression Model')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Actual Close Prices')
plt.ylabel('Predicted Close Prices')
plt.tight_layout()
plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residual Values')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()