import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Load and Prepare the Dataset
df = pd.read_csv('titanic.csv')  # Replace with full path if needed
# Drop irrelevant or high-missing columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Handle missing values
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0],
    'Fare': df['Fare'].median()
}, inplace=True)
# Encode Categorical Features
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = label_enc.fit_transform(df['Embarked']) # S=2, C=0, Q=1
# Feature and Target Split
X = df.drop('Survived', axis=1)
y = df['Survived']
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
# Make Predictions
y_pred = log_model.predict(X_test)
# Evaluate Model Performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Survived', 'Survived'],
    yticklabels=['Not Survived', 'Survived']
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Titanic Logistic Regression')
plt.tight_layout()
plt.show()