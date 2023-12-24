import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('winequality.csv')
    return data

# Main function
def main():
    st.title('Aplikasi Data Mining dengan Metode K-Means')
    st.write('df')

    # Load dataset
    df = load_data()

    # Display dataset
    st.subheader('winequality')
    st.write(df)

    # K-Means clustering
    st.subheader('Hasil Clustering dengan K-Means')
    k = st.slider('Jumlah Cluster (K)', 2, 10, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    df['Cluster'] = kmeans.labels_
    st.write(df)

if __name__ == '__main__':
    main()
    
