# Experiment 6 - Unsupervised Learning (K-Means on Wholesale Customers Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# Step 1: Load Dataset
data = fetch_ucirepo(id=292)
df = pd.concat([data.data.features, data.data.targets], axis=1)
print("Dataset loaded successfully!")
print(df.head())

# Step 2: Select Numerical Features
X = df[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]

# Step 3: Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find Optimal K using Silhouette Score
scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    scores.append(score)
    print(f"k={k}, Silhouette Score={score:.3f}")

optimal_k = np.argmax(scores) + 2
print(f"\nOptimal number of clusters: {optimal_k}")

# Step 5: Apply K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Step 6: Analyze Clusters
print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

print("\nCluster Centers:")
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                       columns=X.columns)
print(centers.round(0))

# Step 7: Visualize Using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['Cluster'], palette='Set1')
plt.title("Customer Segments (PCA Visualization)")
plt.show()
