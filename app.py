# app.py — K-Means con Streamlit (parámetros configurables)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="K-Means", page_icon="🧩", layout="centered")
st.title("Aprendizaje no supervisado: k-means")
st.subheader("By Axel Mireles ITC")

# --------- Cargar datos ----------
st.markdown("### cargar datos")
archivo = st.file_uploader("Sube un archivo CSV con tus datos (al menos 2 columnas numéricas)", type=["csv"])
if archivo is None:
    # usa el que pongas en el repo
    df = pd.read_csv("analisis.csv")
    st.info("Usando el archivo por defecto: **analisis.csv**")
else:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")

# --------- Columnas numéricas ----------
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(num_cols) < 2:
    st.error("Tu CSV debe tener **al menos dos columnas numéricas**.")
    st.stop()

# Heurística para elegir dos columnas “comunes”
def pick(name_lower: str, fallback_idx: int):
    for c in df.columns:
        if c.strip().lower() == name_lower:
            return c
    return num_cols[fallback_idx]

x_def = pick("ingresos", 0)
y_def = pick("puntuacion", 1 if len(num_cols) > 1 else 0)
x_def = pick("saldo", num_cols.index(x_def))
y_def = pick("transacciones", num_cols.index(y_def))

st.markdown("### Datos")
st.dataframe(df[[x_def, y_def]].head(188), use_container_width=True)

# --------- Normalización ----------
X = df[[x_def, y_def]].copy()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)
df_scaled = pd.DataFrame(X_scaled, columns=["Saldo", "transacciones"])
st.dataframe(df_scaled.head(188), use_container_width=True)

# --------- Parámetros ----------
st.sidebar.header("Parámetros del modelo")
k = st.sidebar.slider("k (número de clústeres)", 2, 10, 3, 1)
init = st.sidebar.selectbox("init", options=["k-means++", "random"], index=0)
n_init = st.sidebar.number_input("n_init", min_value=1, value=10, step=1)
max_iter = st.sidebar.number_input("max_iter", min_value=10, value=300, step=10)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

# --------- Entrenar ----------
kmeans = KMeans(
    n_clusters=k,
    init=init,
    n_init=n_init,
    max_iter=max_iter,
    random_state=random_state,
)
labels = kmeans.fit_predict(X_scaled)
inercia = kmeans.inertia_
sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0

cent_scaled = kmeans.cluster_centers_
cent_original = scaler.inverse_transform(cent_scaled)
centros_df = pd.DataFrame(cent_original, columns=[x_def, y_def])
centros_df.index.name = "cluster"

# Mostrar centroides / métricas (como en tu referencia)
st.write(cent_scaled.tolist())
st.write(inercia)

# --------- Gráfica scatter (normalizada) ----------
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#FF6B6B", "#4D96FF", "#FFB86B", "#6BCB77", "#C77DFF", "#FFD166", "#00C2A8", "#9B59B6", "#2ECC71"]
for c in range(k):
    m = (labels == c)
    ax.scatter(df_scaled.iloc[m, 0], df_scaled.iloc[m, 1], s=60, alpha=0.9, color=colors[c % len(colors)])
ax.set_title("clientes")
ax.set_xlabel("saldo en cuenta de ahorros")
ax.set_ylabel("veces que uso tarjeta de credito")
ax.text(1.02, 0.85, f"k={k}", transform=ax.transAxes)
ax.text(1.02, 0.78, f"inercia = {inercia:.2f}", transform=ax.transAxes)
st.pyplot(fig)

# --------- Método del codo ----------
ks = list(range(2, 11))
inercias = []
for kk in ks:
    km = KMeans(n_clusters=kk, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
    km.fit(X_scaled)
    inercias.append(km.inertia_)

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.scatter(ks, inercias, s=60, color="#7E57C2")
ax2.plot(ks, inercias, color="#7E57C2")
ax2.set_xlabel("numero de clusters")
ax2.set_ylabel("inercia")
st.pyplot(fig2)

# --------- Descargar resultado ----------
df_out = df.copy()
df_out["cluster"] = labels
st.download_button(
    "Descargar CSV con clústeres",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="resultado_kmeans.csv",
    mime="text/csv",
)

# Info rápida opcional
with st.expander("Info del modelo"):
    st.write(f"Inercia: {inercia:.4f}")
    st.write(f"Silhouette: {sil:.4f}")
    st.write("Centroides (escala original):")
    st.dataframe(centros_df)
