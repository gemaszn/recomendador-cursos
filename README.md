# Recomendador de cursos

Proyecto desarrollado para el módulo **Sistemas de Aprendizaje Automático**.

El objetivo del proyecto es crear un sistema de recomendación de cursos para una plataforma educativa online. Para ello se utilizan técnicas de aprendizaje no supervisado, clustering y filtrado colaborativo.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- FastAPI
- Streamlit
- Joblib

## Estructura del proyecto

```text
recomendador_cursos/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── src/
│   ├── generar_datos.py
│   ├── entrenar_modelo.py
│   ├── recomendar.py
│   └── evaluar.py
├── requirements.txt
└── README.md
```

## Descripción

El sistema genera un conjunto de datos sintético con interacciones entre usuarios y cursos.
Después, agrupa a los usuarios mediante K-means y utiliza filtrado colaborativo para recomendar cursos a partir de usuarios similares.

El proyecto también incluye:

Entrenamiento del modelo de clustering.
Evaluación mediante índice de Silhouette y Precisión@K.
API con FastAPI.
Interfaz visual con Streamlit.

## Instalación

Crear entorno virtual:
```text
python -m venv venv
```

Activar entorno virtual en Windows:
```text
venv\Scripts\activate
```

Instalar dependencias:
```text
pip install -r requirements.txt
```

## Ejecución

Generar los datos:
```text
python src/generar_datos.py
```

Entrenar el modelo:
```text
python src/entrenar_modelo.py
```

Probar el recomendador:
```text
python src/recomendar.py
```

Evaluar el sistema:
```text
python src/evaluar.py
```

Ejecutar la API:
```text
uvicorn api.main:app --reload
```

La API estará disponible en:
```text
http://127.0.0.1:8000
```

Documentación de la API:
```text
http://127.0.0.1:8000/docs
```

Ejemplo de recomendación:
```text
http://127.0.0.1:8000/recomendar/1?top_n=5
```

Ejecutar la aplicación visual:
```text
streamlit run app/streamlit_app.py
```

La aplicación se abrirá en:
```text
http://localhost:8501
```

## Funcionamiento

El sistema sigue estos pasos:

Genera datos simulados de usuarios y cursos.
Agrupa los usuarios según su comportamiento.
Calcula la similitud entre usuarios del mismo cluster.
Recomienda cursos que usuarios similares han valorado positivamente.
Muestra las recomendaciones mediante API y Streamlit.
Métricas utilizadas
Índice de Silhouette.
Precisión@K.
Limitaciones

El proyecto utiliza datos sintéticos, por lo que los resultados no representan el comportamiento real de estudiantes. En una aplicación real sería recomendable usar datos históricos reales y aplicar una validación más completa.

## Autor
Gema Sánchez Navarro
