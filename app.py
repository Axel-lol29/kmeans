# app.py — Clustering con K-Means y PCA (Comparación antes/después) | Streamlit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Configuración de la página ---
st.set_page_config(page_title="Clustering con K-Means", page_icon="🧩", layout="wide")
st.title("Clustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.subheader("By Axel Mireles ITC")

st.markdown(
    "Sube tus datos, aplica **K-Means**, observa cómo el algoritmo agrupa los puntos en un espacio reducido con PCA (2D o 3D). "
    "También puedes comparar la distribución **antes y después** del clustering."
)

# --- Cargar archivo CSV ---
archivo = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if archivo is None:
    df = pd.read_csv("analisis.csv")
    st.info("Usando el archivo por defecto: **analisis.csv**")
else:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")

# --- Vista previa de los datos ---
st.subheader("Vista previa de los datos")
st.dataframe(df.head(10))

# --- Selección de columnas numéricas y parámetros ---
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
x_col = st.sidebar.selectbox("Selecciona columna X", options=num_cols, index=0)
y_col = st.sidebar.selectbox("Selecciona columna Y", options=num_cols, index=1)

# --- Parámetros de K-Means ---
st.sidebar.header("Configuración del modelo")
k = st.sidebar.slider("Número de clústeres (k)", 2, 10, 3, 1)
init = st.sidebar.selectbox("Método de inicialización (init)", options=["k-means++", "random"], index=0)
n_init = st.sidebar.number_input("n_init", min_value=1, value=10, step=1)
max_iter = st.sidebar.number_input("max_iter", min_value=100, value=300, step=50)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

# --- Selección de visualización PCA ---
visual_pca = st.sidebar.radio("Visualización PCA", options=[2, 3], index=0)

# --- Preprocesamiento de datos ---
X = df[[x_col, y_col]].copy()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)

# --- PCA sin clustering (para visualización antes del clustering) ---
pca = PCA(n_components=visual_pca)
X_pca = pca.fit_transform(X_scaled)

# --- Distribución original (antes de K-Means) ---
st.subheader("Distribución original (antes de K-Means)")
fig1 = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1], color_discrete_sequence=["gray"],
    labels={"x": "PCA1", "y": "PCA2"}, title="Datos originales proyectados con PCA (sin agrupar)"
)
st.plotly_chart(fig1)

# --- K-Means (modelo final) ---
kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
kmeans.fit(X_scaled)

# --- Gráfico de K-Means con PCA ---
labels = kmeans.labels_

st.subheader(f"Datos agrupados con K-Means (k = {k})")
fig2 = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1], color=labels.astype(str), 
    color_discrete_sequence=px.colors.qualitative.Set1,
    labels={"x": "PCA1", "y": "PCA2"},
    title=f"Clústeres visualizados en {visual_pca}D con PCA"
)
st.plotly_chart(fig2)

# --- Centroides de los clústeres ---
cent_scaled = kmeans.cluster_centers_
cent_original = scaler.inverse_transform(cent_scaled)
centros_df = pd.DataFrame(cent_original, columns=[x_col, y_col])
centros_df.index.name = "cluster"

st.subheader("Centroides de los clústeres (en espacio PCA)")
st.dataframe(centros_df)

# --- Método del codo (Elbow Method) ---
st.subheader("Método del Codo (Elbow Method)")
elbow_button = st.button("Calcular número óptimo de clústeres")
if elbow_button:
    ks = list(range(2, 11))
    inercias = []
    for kk in ks:
        km = KMeans(n_clusters=kk, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(ks, inercias, marker="o", color="purple")
    ax3.set_xlabel("Número de clústeres")
    ax3.set_ylabel("Inercia")
    ax3.set_title("Método del Codo (Elbow Method)")
    st.pyplot(fig3)

# --- Descargar CSV con clústeres asignados ---
st.subheader("Descargar datos con clústeres asignados")
st.download_button(
    "Descargar CSV con Clústeres",
    data=df.copy().assign(cluster=labels).to_csv(index=False).encode("utf-8"),
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)
