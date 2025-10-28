import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load and clean datase
df = pd.read_csv("adult.csv")
# Replace '?' with NaN and drop missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
# Encode categorical feature
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
# Feature and target spli
X = df.drop("income", axis=1)
y = df["income"]
# Train-test spli
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Decision Tree Classifie
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
# Model Evaluatio
y_pred = dt_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Decision Tree Visualization (top levels only
plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=X.columns,
    class_names=label_encoders['income'].classes_,
    filled=True,
    max_depth=3,
    fontsize=10
)
plt.title('Decision Tree Visualization (Top Levels)')
plt.show()
# Confusion Matrix Heatma
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoders['income'].classes_,
    yticklabels=label_encoders['income'].classes_
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.tight_layout()
plt.show()