# actividad2.py — K-Means con PCA (con controles solicitados)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# Configuración de la app
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("🎯 lustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.write("""
Axel Mireles: 739047
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
""")

# --- Subir archivo ---
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("⚙️ Configuración del modelo")

        # Seleccionar columnas a usar (igual que el programa original)
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        # Parámetros de clustering
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)

        # 🔽 LO QUE PIDE LA ACTIVIDAD: exponer estos cuatro hiperparámetros
        init_method = st.sidebar.selectbox(
            "init",
            options=["k-means++", "random"],
            index=0,
            help="Método de inicialización de centroides."
        )
        n_init_val = st.sidebar.number_input(
            "n_init",
            min_value=1,
            value=10,
            step=1,
            help="Número de reinicios independientes del algoritmo."
        )
        max_iter_val = st.sidebar.number_input(
            "max_iter",
            min_value=1,
            value=300,
            step=50,
            help="Iteraciones máximas por ejecución."
        )
        random_state_val = st.sidebar.number_input(
            "random_state",
            min_value=0,
            value=0,
            step=1,
            help="Semilla para reproducibilidad."
        )
        # 🔼 FIN de lo solicitado

        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # --- Datos y modelo ---
        X = data[selected_cols]

        # Usar EXACTAMENTE los hiperparámetros seleccionados arriba
        kmeans = KMeans(
            n_clusters=k,
            init=init_method,
            max_iter=int(max_iter_val),
            n_init=int(n_init_val),
            random_state=int(random_state_val),
        )
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        # --- Visualización antes del clustering ---
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- Visualización después del clustering ---
        st.subheader(f"🔷 Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides (en espacio PCA) ---
        st.subheader("🧭 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # --- Método del Codo ---
        st.subheader("📈 Método del Codo (Elbow Method)")
        if st.button("Calcular número óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(
                    n_clusters=i,
                    init=init_method,
                    max_iter=int(max_iter_val),
                    n_init=int(n_init_val),
                    random_state=int(random_state_val),
                )
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(list(K), inertias, 'o-', color='blue')
            ax2.set_title('método del Codo')
            ax2.set_xlabel('Número de Clusters (k)')
            ax2.set_ylabel('Inercia (SSE)')
            ax2.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("💾 Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | ingreso | gasto | edad |
    |---------|-------|------|
    | 45000   | 350   | 28   |
    | 72000   | 680   | 35   |
    | 28000   | 210   | 22   |
    """)
