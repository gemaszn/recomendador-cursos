import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("src"))

from recomendar import recomendar_cursos

df = pd.read_csv("data/interacciones.csv")
usuarios_clusterizados = pd.read_csv("data/usuarios_clusterizados.csv")

st.set_page_config(
    page_title="Recomendador de cursos",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Sistema de recomendación de cursos")
st.write(
    "Aplicación basada en aprendizaje no supervisado, clustering "
    "y filtrado colaborativo."
)

usuario_id = st.selectbox(
    "Selecciona un usuario",
    sorted(df["usuario_id"].unique())
)

top_n = st.slider(
    "Número de recomendaciones",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Generar recomendaciones"):
    recomendaciones = recomendar_cursos(usuario_id, top_n)

    st.subheader(f"Recomendaciones para el usuario {usuario_id}")

    if recomendaciones:
        st.dataframe(pd.DataFrame(recomendaciones))
    else:
        st.warning("No se han encontrado recomendaciones para este usuario.")

    cluster = usuarios_clusterizados[
        usuarios_clusterizados["usuario_id"] == usuario_id
    ]["cluster"].iloc[0]

    st.info(f"El usuario pertenece al cluster {cluster}")

st.subheader("Distribución de usuarios por cluster")

fig, ax = plt.subplots(figsize=(7, 4))
usuarios_clusterizados["cluster"].value_counts().sort_index().plot(
    kind="bar",
    ax=ax
)
ax.set_xlabel("Cluster")
ax.set_ylabel("Número de usuarios")
ax.set_title("Usuarios por cluster")
st.pyplot(fig)

st.subheader("Datos de interacciones")
st.dataframe(df.head(50))