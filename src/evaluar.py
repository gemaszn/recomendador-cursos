import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/interacciones.csv")
usuarios_clusterizados = pd.read_csv("data/usuarios_clusterizados.csv")


def recomendar_cursos_evaluacion(usuario_id, df_train, user_course_matrix, top_n=5):
    """
    Versión del recomendador adaptada para evaluación.
    Usa solo los datos de entrenamiento.
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

    if matriz_cluster.empty:
        return []

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
        df_train[df_train["usuario_id"] == usuario_id]["curso_id"].unique()
    )

    cursos_recomendados = {}

    for usuario_similar in usuarios_similares:
        cursos_similar = df_train[df_train["usuario_id"] == usuario_similar]

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


def crear_train_test(df):
    """
    Oculta un curso bien valorado por usuario para usarlo como prueba.
    """

    train_rows = []
    test_rows = []

    for usuario_id, grupo in df.groupby("usuario_id"):
        cursos_relevantes = grupo[grupo["calificacion"] >= 4]

        if len(cursos_relevantes) >= 1 and len(grupo) > 1:
            test = cursos_relevantes.sample(1, random_state=42)
            train = grupo.drop(test.index)

            train_rows.append(train)
            test_rows.append(test)
        else:
            train_rows.append(grupo)

    df_train = pd.concat(train_rows)

    if test_rows:
        df_test = pd.concat(test_rows)
    else:
        df_test = pd.DataFrame(columns=df.columns)

    return df_train, df_test


def precision_at_k(k=5):
    df_train, df_test = crear_train_test(df)

    user_course_matrix_train = df_train.pivot_table(
        index="usuario_id",
        columns="curso_id",
        values="calificacion",
        fill_value=0
    )

    precisiones = []

    for usuario_id in df_test["usuario_id"].unique():
        recomendaciones = recomendar_cursos_evaluacion(
            usuario_id,
            df_train,
            user_course_matrix_train,
            top_n=k
        )

        cursos_recomendados = [r["curso_id"] for r in recomendaciones]

        cursos_relevantes_test = df_test[
            df_test["usuario_id"] == usuario_id
        ]["curso_id"].unique()

        aciertos = len(set(cursos_recomendados) & set(cursos_relevantes_test))

        precisiones.append(aciertos / k)

    if len(precisiones) == 0:
        return 0

    return sum(precisiones) / len(precisiones)


resultado = precision_at_k(k=5)

print(f"Precisión@5 media: {resultado:.4f}")