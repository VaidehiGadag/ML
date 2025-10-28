import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from ucimlrepo import fetch_ucirepo
# Load dataset
wholesale_customers = fetch_ucirepo(id=292)
X = wholesale_customers.data.features
y = wholesale_customers.data.targets
data = pd.concat([X, y], axis=1)
print("Dataset loaded successfully!")
print(f"Shape: {data.shape}")
# 1. EXPLORATORY DATA ANALYSIS
print("\n=== EXPLORATORY DATA ANALYSIS ===")
print("Dataset Info:")
print(data.info())
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nStatistical Summary:")
print(data.describe())
# 2. FEATURE ENGINEERING
print("\n=== FEATURE ENGINEERING ===")
df_features = data.copy()
label_encoder_channel = LabelEncoder()
label_encoder_region = LabelEncoder()
df_features['Channel_encoded'] = label_encoder_channel.fit_transform(df_features['Channel'])
df_features['Region_encoded'] = label_encoder_region.fit_transform(df_features['Region'])
spending_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
df_features['Total_Spending'] = df_features[spending_features].sum(axis=1)
for feature in spending_features:
    df_features[f'{feature}_Ratio'] = df_features[feature] / df_features['Total_Spending']
df_features['Spending_Category'] = pd.cut(
    df_features['Total_Spending'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)
print("Feature engineering completed!")
print(f"New dataset shape: {df_features.shape}")
# 3. DATA PREPROCESSING FOR CLUSTERING
print("\n=== DATA PREPROCESSING ===")
clustering_data = df_features[spending_features].copy()
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean
clustering_data_final = clustering_data.copy()
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data_final)
clustering_df_scaled = pd.DataFrame(
    clustering_data_scaled,
    columns=spending_features,
    index=clustering_data_final.index
)
print("Data preprocessing completed!")
print(f"Final clustering data shape: {clustering_df_scaled.shape}")
# 4. OPTIMAL NUMBER OF CLUSTERS
print("\n=== FINDING OPTIMAL NUMBER OF CLUSTERS ===")
inertias = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(clustering_df_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(clustering_df_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.3f}")
optimal_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
# 5. APPLY K-MEANS CLUSTERING
print("\n=== APPLYING K-MEANS CLUSTERING ===")
final_k = 3
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(clustering_df_scaled)
# Convert safely to pandas category type
df_features['Cluster'] = pd.Series(cluster_labels, index=df_features.index, dtype='category')
print(f"K-Means clustering applied with k={final_k}")
print(f"Inertia: {kmeans_final.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(clustering_df_scaled, cluster_labels):.3f}")
# 6. CLUSTER ANALYSIS AND INTERPRETATION
print("\n=== CLUSTER ANALYSIS ===")
print("Cluster Distribution:")
print(df_features['Cluster'].value_counts().sort_index())
print("\nCluster Centers (Original Scale):")
cluster_centers_scaled = kmeans_final.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
cluster_centers_df = pd.DataFrame(
    cluster_centers_original,
    columns=spending_features,
    index=[f'Cluster {i}' for i in range(final_k)]
)
print(cluster_centers_df.round(2))
print("\nCluster Characteristics:")
for i in range(final_k):
    cluster_data = df_features[df_features['Cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster_data)}):")
    print(f" Average Total Spending: {cluster_data['Total_Spending'].mean():.0f}")
    print(f" Top spending category: {cluster_data[spending_features].mean().idxmax()}")
    print(f" Channel distribution: {cluster_data['Channel'].value_counts().to_dict()}")
    print(f" Region distribution: {cluster_data['Region'].value_counts().to_dict()}")
# 7. DIMENSIONALITY REDUCTION FOR VISUALIZATION
print("\n=== DIMENSIONALITY REDUCTION ===")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(clustering_df_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels
# 8. VISUALIZATION OF CLUSTERS
print("\n=== VISUALIZATION ===")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_df['PC1'],
    y=pca_df['PC2'],
    hue=pca_df['Cluster'],
    palette='Set1',
    s=60,
    alpha=0.8
)
cluster_centers_pca = pca.transform(cluster_centers_scaled)
plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)
plt.title('Customer Segments (PCA-reduced)', fontsize=14)
plt.legend()
plt.show()
sns.pairplot(df_features[spending_features + ['Cluster']], hue='Cluster', palette='Set1', diag_kind='kde')
plt.suptitle("Pairplot of Spending Features by Cluster", y=1.02)
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers_df.T, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True)
plt.title("Cluster Centers - Spending Patterns", fontsize=14)
plt.ylabel("Features")
plt.xlabel("Clusters")
plt.show()