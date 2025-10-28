import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
# Load and preprocess the dataset
def load_and_preprocess_data(filename='adult.csv'):
    try:
        df = pd.read_csv(filename)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        if df.shape[1] == 15 and 'age' not in df.columns:
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            df.columns = column_names
        # Clean string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).str.strip()
        # Handle missing values
        df = df.replace('?', np.nan)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
# Feature preprocessing
def preprocess_features(df):
    X = df.drop('income', axis=1)
    y = df['income']
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
    print(f"Features shape: {X_encoded.shape}")
    return X_encoded, y_encoded
# Dimensionality reduction methods
def apply_pca(X_train, X_test, n_components):
    max_components = min(X_train.shape[0], X_train.shape[1])
    n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca
def apply_lda(X_train, X_test, y_train):
    max_components = min(X_train.shape[1], len(np.unique(y_train)) - 1)
    lda = LDA(n_components=max_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda
def apply_feature_selection(X_train, X_test, y_train, k, method='chi2'):
    k = min(k, X_train.shape[1])
    if method == 'chi2':
        X_train_pos = X_train - X_train.min() + 1e-8
        X_test_pos = X_test - X_test.min() + 1e-8
        selector = SelectKBest(score_func=chi2, k=k)
        X_train_selected = selector.fit_transform(X_train_pos, y_train)
        X_test_selected = selector.transform(X_test_pos)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected
# Model evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {
            'Model': model_name,
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-Score': 0.0
        }
# Main analysis pipeline
def run_dimensionality_reduction_analysis():
    df = load_and_preprocess_data('adult.csv')
    if df is None:
        return None
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42)
    }
    results = []
    max_features = X_train.shape[1]
    print(f"\n=== ANALYSIS WITH {max_features} FEATURES ===\n")
    # 1. Baseline
    print("1. BASELINE - Original Features")
    for model_name, model in models.items():
        result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test,
                                f"Baseline_{model_name}")
        result['Technique'] = 'Original'
        result['Components'] = max_features
        results.append(result)
    # 2. PCA
    print("\n2. PCA")
    pca_components = sorted(list(set([c for c in [5, 10, max_features - 1, max_features] if c > 0])))
    for n_comp in pca_components:
        X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_comp)
        for model_name, model in models.items():
            result = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test,
                                    f"PCA{n_comp}_{model_name}")
            result['Technique'] = f'PCA-{n_comp}'
            result['Components'] = n_comp
            results.append(result)
    # 3. LDA
    print("\n3. LDA")
    X_train_lda, X_test_lda = apply_lda(X_train_scaled, X_test_scaled, y_train)
    for model_name, model in models.items():
        result = evaluate_model(model, X_train_lda, X_test_lda, y_train, y_test,
                                f"LDA_{model_name}")
        result['Technique'] = 'LDA-1'
        result['Components'] = 1
        results.append(result)
    # 4. Feature Selection
    print("\n4. FEATURE SELECTION")
    feature_counts = sorted(list(set([c for c in [5, 10, max_features - 1, max_features] if c > 0])))
    for k in feature_counts:
        for method in ['chi2', 'f_classif']:
            X_train_sel, X_test_sel = apply_feature_selection(X_train_scaled, X_test_scaled,
                                                              y_train, k, method)
            for model_name, model in models.items():
                result = evaluate_model(model, X_train_sel, X_test_sel, y_train, y_test,
                                        f"{method.upper()}{k}_{model_name}")
                result['Technique'] = f'{method.upper()}-{k}'
                result['Components'] = k
                results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv('dimensionality_reduction_results.csv', index=False)
    print(f"\nResults saved to 'dimensionality_reduction_results.csv'")
    return results_df
# Results analysis
def analyze_results(results_df):
    if results_df is None or results_df.empty:
        return None
    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    print("\n🏆 BEST PERFORMANCE")
    print(f"Best Accuracy: {best_accuracy['Model']} - {best_accuracy['Technique']} "
          f"({best_accuracy['Accuracy']:.4f})")
    print(f"Best F1-Score: {best_f1['Model']} - {best_f1['Technique']} "
          f"({best_f1['F1-Score']:.4f})")
    technique_summary = (
        results_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        .mean()
        .round(4)
    )
    print("\n📊 AVERAGE PERFORMANCE BY TECHNIQUE:")
    print(technique_summary)
    return technique_summary
# Visualization
def visualize_results(results_df):
    if results_df is None or results_df.empty:
        return
    try:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        sns.barplot(data=results_df, x='Technique', y='Accuracy', hue='Model')
        plt.title('Model Accuracy by Dimensionality Reduction Technique')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplot(2, 1, 2)
        sns.barplot(data=results_df, x='Technique', y='F1-Score', hue='Model')
        plt.title('Model F1-Score by Dimensionality Reduction Technique')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
# Main entry point
if __name__ == "__main__":
    print("Starting Dimensionality Reduction Analysis...")
    results = run_dimensionality_reduction_analysis()
    if results is not None:
        analyze_results(results)
        visualize_results(results)
        print("Analysis Complete!")
    else:
        print("Analysis failed. Please check your dataset file.")