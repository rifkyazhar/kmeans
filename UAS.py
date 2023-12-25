import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Algoritma K-means
def kmeans_algorithm(data, k, num_iterations):
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=num_iterations)
    kmeans.fit(data)
    return kmeans

# Visualisasi data
def visualize_data(data, kmeans):
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    fig, ax = plt.subplots()

    for i in range(k):
        xs = data[labels == i, 0]
        ys = data[labels == i, 1]
        plt.scatter(xs, ys, label=f'Cluster {i + 1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Scaled Annual Income')
    plt.ylabel('Scaled Spending Score')
    plt.legend()
    plt.show()

    st.pyplot(fig)


 def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, clusters)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return clusters

    def _initialize_centroids(self, X):
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=-1)
        return np.argmin(distances, axis=-1)

    def _update_centroids(self, X, clusters):
        new_centroids = np.empty_like(self.centroids)

        for i in range(self.k):
            new_centroids[i] = np.mean(X[clusters == i], axis=0)

        return new_centroids



# Aplikasi utama
st.title('K-means Clustering')

data = np.random.rand(500, 2)
k = st.slider('Jumlah cluster', 2, 10, 5)
num_iterations = st.slider('Jumlah iterasi', 10, 100, 50)

if st.button('Lakukan Clustering'):
    kmeans = kmeans_algorithm(data, k, num_iterations)
    visualize_data(data, kmeans)
