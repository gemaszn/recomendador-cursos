from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath("src"))

from recomendar import recomendar_cursos

app = FastAPI(
    title="API de Recomendación de Cursos",
    description="Sistema de recomendación basado en clustering y filtrado colaborativo",
    version="1.0"
)


@app.get("/")
def inicio():
    return {
        "mensaje": "API de recomendación de cursos en funcionamiento"
    }


@app.get("/recomendar/{usuario_id}")
def recomendar(usuario_id: int, top_n: int = 5):
    recomendaciones = recomendar_cursos(usuario_id, top_n)

    return {
        "usuario_id": usuario_id,
        "recomendaciones": recomendaciones
    }