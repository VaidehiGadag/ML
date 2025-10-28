# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Step 2: Load dataset (CSV format)
df = pd.read_csv('BostonHousing.csv')
# Step 3: Explore the dataset
print(df.head())
print(df.info())
print(df.describe())
# Optional: Check correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
# Step 4: Handle missing values
df.dropna(inplace=True)  # Drop rows with missing values for simplicity
# Step 5: Define features and target
X = df.drop('medv', axis=1)  # medv = median house value (target)
y = df['medv']
# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 7: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 8: Make predictions
y_pred = model.predict(X_test)
# Step 9: Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# Step 10: Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()
# Step 11: Display model coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)