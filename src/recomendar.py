import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/interacciones.csv")
usuarios_clusterizados = pd.read_csv("data/usuarios_clusterizados.csv")
user_course_matrix = joblib.load("models/user_course_matrix.pkl")


def recomendar_cursos(usuario_id: int, top_n: int = 5):
    """
    Recomienda cursos a un usuario mediante filtrado colaborativo
    basado en usuarios similares dentro del mismo cluster.
    """

    if usuario_id not in user_course_matrix.index:
        return []

    usuario_info = usuarios_clusterizados[
        usuarios_clusterizados["usuario_id"] == usuario_id
    ]

    if usuario_info.empty:
        return []

    cluster_usuario = usuario_info["cluster"].iloc[0]

    usuarios_mismo_cluster = usuarios_clusterizados[
        usuarios_clusterizados["cluster"] == cluster_usuario
    ]["usuario_id"].tolist()

    matriz_cluster = user_course_matrix.loc[
        user_course_matrix.index.intersection(usuarios_mismo_cluster)
    ]

    usuario_vector = user_course_matrix.loc[[usuario_id]]

    similitudes = cosine_similarity(usuario_vector, matriz_cluster)[0]

    similitudes_df = pd.DataFrame({
        "usuario_id": matriz_cluster.index,
        "similitud": similitudes
    })

    similitudes_df = similitudes_df[
        similitudes_df["usuario_id"] != usuario_id
    ].sort_values(by="similitud", ascending=False)

    usuarios_similares = similitudes_df.head(5)["usuario_id"].tolist()

    cursos_usuario = set(
        df[df["usuario_id"] == usuario_id]["curso_id"].unique()
    )

    cursos_recomendados = {}

    for usuario_similar in usuarios_similares:
        cursos_similar = df[df["usuario_id"] == usuario_similar]

        for _, row in cursos_similar.iterrows():
            curso = row["curso_id"]

            if curso not in cursos_usuario:
                if curso not in cursos_recomendados:
                    cursos_recomendados[curso] = []

                cursos_recomendados[curso].append(row["calificacion"])

    ranking = []

    for curso, calificaciones in cursos_recomendados.items():
        ranking.append({
            "curso_id": int(curso),
            "puntuacion_media": round(float(np.mean(calificaciones)), 2),
            "num_recomendaciones": len(calificaciones)
        })

    ranking = sorted(
        ranking,
        key=lambda x: (x["puntuacion_media"], x["num_recomendaciones"]),
        reverse=True
    )

    return ranking[:top_n]


if __name__ == "__main__":
    usuario = 1
    recomendaciones = recomendar_cursos(usuario, top_n=5)

    print(f"Recomendaciones para el usuario {usuario}:")
    for rec in recomendaciones:
        print(rec)