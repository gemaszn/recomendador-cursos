import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/interacciones.csv")

# Crear variables agregadas por usuario
usuarios = df.groupby("usuario_id").agg({
    "progreso": "mean",
    "calificacion": "mean",
    "sesiones": "sum",
    "duracion_media": "mean",
    "curso_id": "count"
}).reset_index()

usuarios = usuarios.rename(columns={
    "curso_id": "num_cursos_interactuados"
})

features = [
    "progreso",
    "calificacion",
    "sesiones",
    "duracion_media",
    "num_cursos_interactuados"
]

X = usuarios[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Buscar mejor K con Silhouette
resultados = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    resultados.append((k, score))

mejor_k = max(resultados, key=lambda x: x[1])[0]
mejor_score = max(resultados, key=lambda x: x[1])[1]

print("Resultados Silhouette:")
for k, score in resultados:
    print(f"K={k} -> Silhouette={score:.4f}")

print(f"Mejor K: {mejor_k}")
print(f"Mejor Silhouette: {mejor_score:.4f}")

# Entrenar K-means final
kmeans_final = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
usuarios["cluster"] = kmeans_final.fit_predict(X_scaled)

# Comparación con DBSCAN
dbscan = DBSCAN(eps=1.3, min_samples=4)
usuarios["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

# Crear matriz usuario-curso con calificaciones
user_course_matrix = df.pivot_table(
    index="usuario_id",
    columns="curso_id",
    values="calificacion",
    fill_value=0
)

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans_final, "models/kmeans.pkl")
joblib.dump(user_course_matrix, "models/user_course_matrix.pkl")
usuarios.to_csv("data/usuarios_clusterizados.csv", index=False)

# Gráfico de Silhouette
ks = [r[0] for r in resultados]
scores = [r[1] for r in resultados]

plt.figure(figsize=(8, 5))
plt.plot(ks, scores, marker="o")
plt.xlabel("Número de clusters K")
plt.ylabel("Índice de Silhouette")
plt.title("Optimización de K-means mediante Silhouette")
plt.grid(True)
plt.savefig("data/silhouette_kmeans.png")
plt.show()

print("Modelo entrenado y guardado correctamente.")