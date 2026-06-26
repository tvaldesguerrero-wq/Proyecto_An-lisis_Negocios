"""
utils/model_utils.py
Funciones encargadas de cargar el modelo y realizar las predicciones.

ESTADO ACTUAL: usa lógica DUMMY (simulada) para que la app funcione de
extremo a extremo mientras el equipo de modelamiento entrena el modelo real.

CUANDO EL MODELO REAL ESTÉ LISTO, reemplazar así:

    import pickle
    import os

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_ventas.pkl")

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    def predecir_ventas(valores_formulario):
        # Convertir valores_formulario al mismo formato/encoding usado en
        # el entrenamiento (mismas columnas, mismo orden, mismo escalado).
        entrada = preparar_entrada(valores_formulario)
        prediccion = modelo.predict(entrada)
        return round(float(prediccion[0]), 2)

La función obtener_estadisticas_dataset() puede mantenerse igual, leyendo
desde data/ventas_videojuegos.csv con pandas, o conectarse a estadísticas
reales calculadas en el notebook.
"""

import random


def predecir_ventas(valores_formulario):
    """
    Simula una predicción de ventas globales (en millones de copias).

    Recibe un diccionario con las claves: plataforma, genero, anio, critic_score.
    Por ahora devuelve un número pseudo-aleatorio pero estable, solo para
    poder probar el flujo completo del formulario -> resultado.

    Reemplazar el cuerpo de esta función por la predicción real del modelo.
    """
    base = 1.5

    genero = (valores_formulario.get("genero") or "").lower()
    if genero in ("sports", "shooter", "action"):
        base += 2.0
    elif genero in ("puzzle", "strategy"):
        base += 0.3

    try:
        critic_score = float(valores_formulario.get("critic_score") or 70)
        base += (critic_score - 50) / 25
    except ValueError:
        pass

    # Variación pequeña para que no sea un número estático fijo
    random.seed(str(valores_formulario))
    ruido = random.uniform(-0.3, 0.3)

    estimacion = max(0.05, base + ruido)
    return round(estimacion, 2)


def obtener_estadisticas_dataset():
    """
    Devuelve estadísticas resumidas para mostrar en la página principal.

    DUMMY por ahora. Cuando el EDA esté terminado, esto se puede reemplazar
    por una lectura real de data/ventas_videojuegos.csv con pandas, por ejemplo:

        import pandas as pd
        df = pd.read_csv("data/ventas_videojuegos.csv")
        return {
            "total_juegos": len(df),
            "plataformas": df["Platform"].nunique(),
            "generos": df["Genre"].nunique(),
            "anio_min": int(df["Year_of_Release"].min()),
            "anio_max": int(df["Year_of_Release"].max()),
        }
    """
    return {
        "total_juegos": 16720,
        "plataformas": 31,
        "generos": 12,
        "anio_min": 1980,
        "anio_max": 2020,
    }
