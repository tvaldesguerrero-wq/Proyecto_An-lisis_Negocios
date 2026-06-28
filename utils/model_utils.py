"""
utils/model_utils.py
Funciones encargadas de cargar el modelo y realizar las predicciones.

Carga el modelo real (Random Forest) entrenado en el notebook del proyecto,
exportado con pickle, y expone también su feature importance para
mostrarla en la interfaz.
"""

import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_ventas.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columnas_modelo.pkl")

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    columnas_modelo = pickle.load(f)

# Valores por defecto para variables que el formulario no solicita,
# correspondientes a las medianas calculadas durante la limpieza (df_limpio).
DEFAULT_CRITIC_COUNT = 21.0
DEFAULT_USER_SCORE = 7.5
DEFAULT_USER_COUNT = 24.0
DEFAULT_RATING = "Sin clasificar"
DEFAULT_PUBLISHER = "Otros"

# Etiquetas legibles para las variables más técnicas del feature importance
ETIQUETAS_VARIABLES = {
    "User_Count": "Cantidad de reseñas de usuarios",
    "Year_of_Release": "Año de lanzamiento",
    "Critic_Score": "Puntaje de la crítica",
    "Critic_Count": "Cantidad de reseñas de críticos",
    "User_Score": "Puntaje de usuarios",
}


def _etiqueta_legible(nombre_columna):
    """Convierte nombres de columnas dummy en etiquetas legibles para la UI."""
    if nombre_columna in ETIQUETAS_VARIABLES:
        return ETIQUETAS_VARIABLES[nombre_columna]
    if nombre_columna.startswith("Platform_"):
        return f"Plataforma: {nombre_columna.replace('Platform_', '')}"
    if nombre_columna.startswith("Genre_"):
        return f"Género: {nombre_columna.replace('Genre_', '')}"
    if nombre_columna.startswith("Publisher_"):
        return f"Publisher: {nombre_columna.replace('Publisher_', '')}"
    if nombre_columna.startswith("Rating_"):
        return f"Clasificación: {nombre_columna.replace('Rating_', '')}"
    return nombre_columna


def construir_entrada(valores_formulario):
    """
    Construye un DataFrame de una sola fila, con las mismas columnas
    (y mismo orden) que el dataset usado para entrenar el modelo.
    """
    fila = {
        "Year_of_Release": float(valores_formulario.get("anio") or 2020),
        "Critic_Score": float(valores_formulario.get("critic_score") or 70),
        "Critic_Count": DEFAULT_CRITIC_COUNT,
        "User_Score": DEFAULT_USER_SCORE,
        "User_Count": DEFAULT_USER_COUNT,
    }

    df_entrada = pd.DataFrame([fila])
    df_entrada["Platform"] = valores_formulario.get("plataforma")
    df_entrada["Genre"] = valores_formulario.get("genero")
    df_entrada["Rating"] = DEFAULT_RATING
    df_entrada["Publisher"] = DEFAULT_PUBLISHER

    columnas_categoricas = ["Platform", "Genre", "Publisher", "Rating"]
    # IMPORTANTE: sin drop_first aquí. Esta función siempre recibe una
    # sola fila, por lo que cada columna categórica tiene una única
    # categoría; drop_first la eliminaría como "base", perdiendo la
    # selección del usuario. El reindex() de abajo alinea correctamente
    # con las columnas reales generadas en el entrenamiento.
    df_entrada = pd.get_dummies(df_entrada, columns=columnas_categoricas, dtype=int)
    df_entrada = df_entrada.reindex(columns=columnas_modelo, fill_value=0)

    return df_entrada


def predecir_ventas(valores_formulario):
    """Genera la predicción de ventas globales (millones de copias)."""
    entrada = construir_entrada(valores_formulario)
    prediccion = modelo.predict(entrada)
    return round(float(prediccion[0]), 2)


def obtener_estadisticas_dataset():
    """Estadísticas generales para la página principal."""
    return {
        "total_juegos": 16416,
        "plataformas": 31,
        "generos": 12,
        "r2_modelo": 0.42,
    }


def obtener_feature_importance(top_n=8):
    """
    Devuelve las top_n variables más importantes del modelo, con
    etiquetas legibles y el valor normalizado a porcentaje (0-100)
    sobre el subconjunto mostrado, para graficarlas como barras.
    """
    pares = list(zip(columnas_modelo, modelo.feature_importances_))
    pares.sort(key=lambda x: x[1], reverse=True)
    top = pares[:top_n]

    maximo = top[0][1] if top else 1
    resultado = [
        {
            "variable": _etiqueta_legible(nombre),
            "importancia": round(float(valor), 4),
            "porcentaje_relativo": round(float(valor) / maximo * 100, 1),
        }
        for nombre, valor in top
    ]
    return resultado
