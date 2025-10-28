# ------------------------------
# Boston Housing Price Prediction using Linear Regression
# ------------------------------

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("BostonHousing.csv")

# Step 3: Basic Info
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Handle Missing Values
df = df.dropna()

# Step 5: Check Correlation (Feature Relation)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 6: Split Data
X = df.drop("medv", axis=1)
y = df["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict & Evaluate
y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Actual vs Predicted Graph
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 10: Coefficients
coef = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print("\nModel Coefficients:\n", coef)
