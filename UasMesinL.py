import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

df = pd.read_csv('winequality.csv')

X = df

df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df.rename(columns={'quality': 'good-quality'}, inplace=True)

st.header("Dataset")
st.write(df)

cluster=[]
for i in range(1,12):
    km =KMeans(n_clusters=i).fit(df)
    cluster.append(km.inertia_)

fig, ax = plt.subplots(figsize=(13,9))
sns.lineplot(x=list(range(1,12)), y=cluster, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('cluster')
ax.set_ylabel('inertia')

ax.annotate('possible elbow point', xy=(2, 650000), xytext=(2, 50000),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2))

ax.annotate('possible elbow point', xy=(4, 300000), xytext=(4, 950000),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,11,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X['volatile acidity'], y=X['alcohol'], hue=X['Labels'], markers=True, size=X['Labels'], palette=sns.color_palette('hls', n_clust))


    for label in X['Labels']:
     plt.annotate(label,
               (X[X['Labels']==label]['volatile acidity'].mean(),
                X[X['Labels']==label]['alcohol'].mean()),
                horizontalalignment = 'center',
                verticalalignment = 'center',
                size = 27, weight='bold',
                color = 'black')
    
    st.header('Cluster plot')
    st.pyplot()
    st.write(X)

k_means(clust)