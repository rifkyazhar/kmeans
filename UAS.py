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
    plt.ylabel…
