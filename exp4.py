import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
# === STEP 1: LOAD DATASET ===
def load_adult_dataset():
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    loading_strategies = [{'file': 'adult.csv', 'names': None, 'header': 0}]
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
            if len(df.columns) == 15 and len(columns) == 15:
                df.columns = columns
                print(f"Dataset loaded successfully from {strategy['file']}!")
                return df
        except FileNotFoundError:
            print(f"File {strategy['file']} not found.")
        except Exception as e:
            print(f"Error loading {strategy['file']}: {e}")
    print("No valid dataset found.")
    return None
print("=== LOADING ADULT CENSUS INCOME DATASET ===")
df = load_adult_dataset()
print(f"Dataset shape: {df.shape}")
print("\n=== DATASET OVERVIEW ===")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget Variable Distribution:")
print(df['income'].value_counts())
# === STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING ===
print("\n=== DATA PREPROCESSING & FEATURE ENGINEERING ===")
print("Handling missing values...")
categorical_columns = ['workclass', 'occupation', 'native-country']
for col in categorical_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
print("Performing feature engineering...")
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly']
)
education_mapping = {
    'Preschool': 'Low', '1st-4th': 'Low', '5th-6th': 'Low', '7th-8th': 'Low',
    '9th': 'Medium', '10th': 'Medium', '11th': 'Medium', '12th': 'Medium',
    'HS-grad': 'Medium', 'Some-college': 'High', 'Assoc-acdm': 'High',
    'Assoc-voc': 'High', 'Bachelors': 'Very High', 'Prof-school': 'Very High',
    'Masters': 'Very High', 'Doctorate': 'Very High'
}
df['education_level'] = df['education'].map(education_mapping)
df['capital_net'] = df['capital-gain'] - df['capital-loss']
df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)
df['work_hours_category'] = pd.cut(
    df['hours-per-week'],
    bins=[0, 20, 40, 60, 100],
    labels=['Part-time', 'Full-time', 'Overtime', 'Extreme']
)
df['is_us_native'] = (df['native-country'] == 'United-States').astype(int)
print("Encoding categorical variables...")
label_encoders = {}
categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'age_group', 'education_level',
    'work_hours_category'
]
for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le
target_encoder = LabelEncoder()
df['income_encoded'] = target_encoder.fit_transform(df['income'])
print("Selecting features for modeling...")
feature_columns = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'capital_net', 'has_capital_gain', 'has_capital_loss', 'is_us_native'
] + [col for col in df.columns if col.endswith('_encoded')]
feature_columns = [col for col in feature_columns if 'income' not in col]
X = df[feature_columns]
y = df['income_encoded']
print(f"Final feature set shape: {X.shape}")
print(f"Features used: {len(X.columns)} features")
# === STEP 3: SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
# === STEP 4: BASELINE MODEL ===
print("\n=== BASELINE RANDOM FOREST MODEL ===")
rf_baseline = RandomForestClassifier(random_state=42, n_estimators=100)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)
print("Baseline Results:")
print(f" Accuracy: {baseline_accuracy:.4f}")
print(f" Precision: {baseline_precision:.4f}")
print(f" Recall: {baseline_recall:.4f}")
print(f" F1-Score: {baseline_f1:.4f}")
# === STEP 5: FEATURE SELECTION ===
print("\n=== FEATURE SELECTION ===")
selector = SelectFromModel(rf_baseline, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features from {len(X.columns)} total features")
print(f"Selected features: {selected_features.tolist()}")
# === STEP 6: HYPERPARAMETER TUNING ===
print("\n=== HYPERPARAMETER TUNING ===")
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
}
print("Performing randomized search for hyperparameter tuning...")
rf_random = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)
rf_random.fit(X_train_selected, y_train)
print(f"Best parameters: {rf_random.best_params_}")
print(f"Best cross-validation score: {rf_random.best_score_:.4f}")
# === STEP 7: FINAL MODEL EVALUATION ===
print("\n=== FINAL MODEL EVALUATION ===")
best_rf = rf_random.best_estimator_
y_pred_best = best_rf.predict(X_test_selected)
y_pred_proba_best = best_rf.predict_proba(X_test_selected)[:, 1]
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)
roc_auc = roc_auc_score(y_test, y_pred_proba_best)
print("=== OPTIMIZED MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_best, target_names=['<=50K', '>50K']))
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
# === STEP 8: FEATURE IMPORTANCE ANALYSIS ===
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("Top 10 Most Important Features:")
print(feature_importance.head(10))
# === CROSS-VALIDATION ===
cv_scores = cross_val_score(best_rf, X_train_selected, y_train, cv=5, scoring='accuracy')
print("\n=== CROSS-VALIDATION RESULTS ===")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
# === MODEL COMPARISON ===
print("\n=== MODEL COMPARISON ===")
print(f"{'Metric':<12} {'Baseline':<10} {'Optimized':<10} {'Improvement':<12}")
print("-" * 50)
print(f"{'Accuracy':<12} {baseline_accuracy:<10.4f} {accuracy:<10.4f} {accuracy - baseline_accuracy:<12.4f}")
print(f"{'Precision':<12} {baseline_precision:<10.4f} {precision:<10.4f} {precision - baseline_precision:<12.4f}")
print(f"{'Recall':<12} {baseline_recall:<10.4f} {recall:<10.4f} {recall - baseline_recall:<12.4f}")
print(f"{'F1-Score':<12} {baseline_f1:<10.4f} {f1:<10.4f} {f1 - baseline_f1:<12.4f}")
# === VISUALIZATIONS ===
print("\n=== CREATING VISUALIZATIONS ===")
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - Optimized Random Forest')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
# 2. Feature Importance
top_features = feature_importance.head(10)
axes[0, 1].barh(range(len(top_features)), top_features['importance'])
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['feature'])
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 10 Feature Importances')
# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend(loc="lower right")
# 4. Model Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
optimized_scores = [accuracy, precision, recall, f1]
x = np.arange(len(metrics))
width = 0.35
axes[1, 1].bar(x - width / 2, baseline_scores, width, label='Baseline', alpha=0.7)
axes[1, 1].bar(x + width / 2, optimized_scores, width, label='Optimized', alpha=0.7)
axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].set_ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig('random_forest_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
# === SUMMARY ===
print("\n" + "=" * 60)
print("RANDOM FOREST ADULT CENSUS INCOME ANALYSIS COMPLETE")
print("=" * 60)
print(f"Dataset Size: {len(df):,} samples")
print(f"Original Features: {len(feature_columns)} features")
print(f"Selected Features: {len(selected_features)} features")
print(f"Train/Test Split: {len(X_train)}/{len(X_test)} samples")
print("\nFINAL PERFORMANCE:")
print(f" • Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f" • Precision: {precision:.4f}")
print(f" • Recall: {recall:.4f}")
print(f" • F1-Score: {f1:.4f}")
print(f" • ROC-AUC: {roc_auc:.4f}")
print("\nIMPROVEMENT OVER BASELINE:")
print(f" • Accuracy: +{accuracy - baseline_accuracy:.4f}")
print(f" • Precision: +{precision - baseline_precision:.4f}")
print(f" • Recall: +{recall - baseline_recall:.4f}")
print(f" • F1-Score: +{f1 - baseline_f1:.4f}")
print("\nAnalysis complete! Results saved to 'random_forest_analysis.png'")