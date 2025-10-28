import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')
# STEP 1: Robust Dataset Loader
def load_adult_dataset():
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    loading_strategies = [
        {'file': 'adult.csv', 'names': None, 'header': 0},
    ]
    for strategy in loading_strategies:
        try:
            if strategy['names']:
                df = pd.read_csv(
                    strategy['file'],
                    names=strategy['names'],
                    na_values=[' ?', '?'],
                    skipinitialspace=True,
                    header=strategy['header']
                )
            else:
                df = pd.read_csv(
                    strategy['file'],
                    na_values=[' ?', '?'],
                    skipinitialspace=True,
                    header=strategy['header']
                )
            if len(df.columns) == 15:
                df.columns = columns
            if df.shape[1] == 15 and 'income' in df.columns:
                print(f"Dataset loaded successfully from {strategy['file']}!")
                print(f"Dataset shape: {df.shape}")
                return df
            else:
                print(f"Invalid dataset format from {strategy['file']}")
        except FileNotFoundError:
            print(f"File {strategy['file']} not found.")
        except Exception as e:
            print(f"Error loading {strategy['file']}: {e}")
    print("No dataset could be loaded with any strategy.")
    return None
# STEP 2: Preprocess Data
def preprocess_adult_dataset(df):
    if df is None:
        return None, None, None, None
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    df = df.dropna()
    print(f"Dataset shape after removing missing values: {df.shape}")
    df['income'] = df['income'].astype(str).str.strip()
    valid_income_values = ['<=50K', '>50K']
    df = df[df['income'].isin(valid_income_values)]
    print(f"Dataset shape after cleaning target: {df.shape}")
    X = df.drop('income', axis=1)
    y = df['income']
    numerical_columns = [
        'age', 'fnlwgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    for col in numerical_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    label_encoder_target = LabelEncoder()
    y_encoded = label_encoder_target.fit_transform(y)
    X_encoded = X.copy()
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        X_encoded[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    print(f"\nTarget distribution: {np.bincount(y_encoded)}")
    print(f"Target classes: {label_encoder_target.classes_}")
    return X_encoded, y_encoded, label_encoder_target, label_encoders
# STEP 3: Evaluation Function
def evaluate_model(model, X_test, y_test, model_name, target_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    print(f"\nConfusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred))
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
# STEP 4: Main Pipeline
def main():
    print("=== LOADING ADULT CENSUS INCOME DATASET ===")
    df = load_adult_dataset()
    if df is None:
        print("Dataset loading failed. Exiting.")
        return
    print("\n=== PREPROCESSING DATASET ===")
    X, y, target_encoder, feature_encoders = preprocess_adult_dataset(df)
    if X is None:
        print("Preprocessing failed. Exiting.")
        return
    print("\n=== SPLITTING DATA ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = []
    # AdaBoost Basic
    print("\n================ AdaBoost Basic ================")
    ada_basic = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada_basic.fit(X_train_scaled, y_train)
    results.append(evaluate_model(ada_basic, X_test_scaled, y_test, "AdaBoost Basic", target_encoder))
    # AdaBoost Hyperparameter Tuning
    print("\n================ AdaBoost Hyperparam Grid ================")
    ada_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0],
        'estimator__max_depth': [1, 2, 3]
    }
    ada_grid = GridSearchCV(
        AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42),
        ada_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    ada_grid.fit(X_train_scaled, y_train)
    print("Best AdaBoost params:", ada_grid.best_params_)
    ada_best = ada_grid.best_estimator_
    results.append(evaluate_model(ada_best, X_test_scaled, y_test, "AdaBoost Optimized", target_encoder))
    # Gradient Boosting Basic
    print("\n================ Gradient Boosting Basic ================")
    gb_basic = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_basic.fit(X_train_scaled, y_train)
    results.append(evaluate_model(gb_basic, X_test_scaled, y_test, "Gradient Boosting Basic", target_encoder))
    # Gradient Boosting Hyperparameter Tuning
    print("\n================ Gradient Boosting Hyperparam Grid ================")
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    gb_grid.fit(X_train_scaled, y_train)
    print("Best Gradient Boosting params:", gb_grid.best_params_)
    gb_best = gb_grid.best_estimator_
    results.append(evaluate_model(gb_best, X_test_scaled, y_test, "Gradient Boosting Optimized", target_encoder))
    # Model Comparison
    print("\n================ Model Comparison ================")
    comparison = pd.DataFrame(results)
    print(comparison[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']].round(4))
    idx = comparison['accuracy'].idxmax()
    print(f"\nBest Model: {comparison.iloc[idx]['model_name']}")
    print(f"Best Accuracy: {comparison.iloc[idx]['accuracy']:.4f}")
    print(f"Best Precision: {comparison.iloc[idx]['precision']:.4f}")
    print(f"Best Recall: {comparison.iloc[idx]['recall']:.4f}")
    print(f"Best F1-Score: {comparison.iloc[idx]['f1_score']:.4f}")
# Run Main
if __name__ == "__main__":
    main()