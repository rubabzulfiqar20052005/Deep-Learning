import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation using Unsupervised Learning")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv","xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------
    # DATA PREPROCESSING
    # ---------------------

    data = df.copy()

    if "Gender" in data.columns:
        if data["Gender"].isnull().sum() == len(data):
            data = data.drop("Gender", axis=1)

    for col in data.columns:

        if data[col].dtype == "object":

            le = LabelEncoder()

            data[col] = le.fit_transform(
                data[col].astype(str)
            )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    # ---------------------
    # PCA
    # ---------------------

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(
        scaled_data
    )

    st.subheader("PCA Visualization")

    fig, ax = plt.subplots()

    ax.scatter(
        pca_data[:,0],
        pca_data[:,1]
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    st.pyplot(fig)

    # ---------------------
    # KMEANS
    # ---------------------

    st.subheader("K-Means")

    k = st.slider(
        "Select Number of Clusters",
        2,
        10,
        5
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    k_labels = kmeans.fit_predict(
        pca_data
    )

    k_score = silhouette_score(
        pca_data,
        k_labels
    )

    # ---------------------
    # HIERARCHICAL
    # ---------------------

    hc = AgglomerativeClustering(
        n_clusters=k
    )

    hc_labels = hc.fit_predict(
        pca_data
    )

    hc_score = silhouette_score(
        pca_data,
        hc_labels
    )

    # ---------------------
    # DBSCAN
    # ---------------------

    eps = st.slider(
        "DBSCAN Epsilon",
        0.1,
        3.0,
        0.6
    )

    dbscan = DBSCAN(
        eps=eps,
        min_samples=5
    )

    db_labels = dbscan.fit_predict(
        pca_data
    )

    if len(set(db_labels)) > 1:
        db_score = silhouette_score(
            pca_data,
            db_labels
        )
    else:
        db_score = -1

    # ---------------------
    # VISUALIZATION
    # ---------------------

    st.subheader(
        "Cluster Comparison"
    )

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15,5)
    )

    ax[0].scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=k_labels
    )
    ax[0].set_title("K-Means")

    ax[1].scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=hc_labels
    )
    ax[1].set_title("Hierarchical")

    ax[2].scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=db_labels
    )
    ax[2].set_title("DBSCAN")

    st.pyplot(fig)

    # ---------------------
    # SCORE TABLE
    # ---------------------

    st.subheader("Model Comparison")

    summary = pd.DataFrame({

        "Algorithm":[
            "KMeans",
            "Hierarchical",
            "DBSCAN"
        ],

        "Silhouette Score":[
            round(k_score,3),
            round(hc_score,3),
            round(db_score,3)
        ],

        "Clusters Found":[
            len(np.unique(k_labels)),
            len(np.unique(hc_labels)),
            len(set(db_labels)) -
            (1 if -1 in db_labels else 0)
        ],

        "Noise Points":[
            0,
            0,
            sum(db_labels == -1)
        ]

    })

    st.dataframe(summary)

    # ---------------------
    # BEST MODEL
    # ---------------------

    best_model = summary.loc[
        summary["Silhouette Score"].idxmax()
    ]

    st.success(
        f"Best Algorithm: {best_model['Algorithm']}"
    )