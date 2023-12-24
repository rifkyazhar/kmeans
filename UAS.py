import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

df = pd.read_csv("winequality.csv")

st.write('STREAMLIT Wine (M)')

x.columns=['Acidity','Sugar' ,'Alcohol']

y=pd.DataFrame(winequality.target)
y.columns=["quality"]

nb=st.slider("Clusters",min_value=2,max_value=x.shape[0],value=2)

st.dataframe(x.head(nb))

nbclust=st.slider("Visualisasi model KMeans clustering",min_value=1,max_value=4,value=2)
model=KMeans(n_clusters=nbclust)
model.fit(x)

fig, ax=plt.subplots()

colormap=np.array(['Red','green','blue',"black"])
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[y.Targets],s=40)
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
st.pyplot(fig)
