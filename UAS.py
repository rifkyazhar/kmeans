import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

try:
    # Blok kode yang mungkin menyebabkan kesalahan
    # ...
    # ...

except Exception as e:
    # Tangkap kesalahan dan tampilkan pesan kesalahan yang aman
    st.error("Aplikasi mengalami kesalahan. Silakan cek log untuk informasi lebih lanjut.")
    # Catat pesan kesalahan lengkap di log
    st.caching.clear_cache()  # Bersihkan cache jika diperlukan
    st.experimental_rerun()  # Rerun aplikasi untuk menghindari kesalahan berulang



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
    
