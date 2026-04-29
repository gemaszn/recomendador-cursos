import pandas as pd
import numpy as np
import os

np.random.seed(42)

usuarios = range(1, 101)
cursos = range(1, 31)

categorias = [
    "Python", "Machine Learning", "Big Data",
    "Bases de Datos", "Cloud", "IA Generativa"
]

datos = []

for usuario in usuarios:
    intereses_usuario = np.random.choice(categorias, size=2, replace=False)

    for curso in cursos:
        categoria = np.random.choice(categorias)

        probabilidad_interaccion = 0.35 if categoria in intereses_usuario else 0.12

        if np.random.rand() < probabilidad_interaccion:
            progreso = np.random.randint(10, 101)
            calificacion = np.random.randint(1, 6)
            sesiones = np.random.randint(1, 15)
            duracion_media = np.random.randint(5, 90)

            datos.append({
                "usuario_id": usuario,
                "curso_id": curso,
                "categoria": categoria,
                "progreso": progreso,
                "calificacion": calificacion,
                "sesiones": sesiones,
                "duracion_media": duracion_media
            })

df = pd.DataFrame(datos)

os.makedirs("data", exist_ok=True)
df.to_csv("data/interacciones.csv", index=False)

print("Dataset generado correctamente.")
print(df.head())
print("Número de registros:", len(df))