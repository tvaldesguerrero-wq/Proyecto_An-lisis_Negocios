"""
utils/model_utils.py
Funciones encargadas de cargar el modelo y realizar las predicciones.

VERSIÓN FINAL: usa el modelo real (Random Forest) entrenado en el notebook
EDA_Proyecto.ipynb / BAGGING.ipynb, exportado con pickle.

Requiere dos archivos en la carpeta models/:
  - modelo_ventas.pkl     -> el modelo de RandomForestRegressor entrenado
  - columnas_modelo.pkl   -> la lista de columnas (con dummies) usada en el entrenamiento
"""

import os
import pickle
import pandas as pd

# ------------------------------------------------------------------
# Carga del modelo y las columnas de referencia (una sola vez al iniciar la app)
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_ventas.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columnas_modelo.pkl")

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    columnas_modelo = pickle.load(f)

# ------------------------------------------------------------------
# Valores por defecto para variables que el formulario no solicita.
# Corresponden a las medianas calculadas en el notebook durante la
# etapa de limpieza (df_limpio), para mantener coherencia con el
# preprocesamiento usado en el entrenamiento.
# ------------------------------------------------------------------

DEFAULT_CRITIC_COUNT = 21.0
DEFAULT_USER_SCORE = 7.5
DEFAULT_USER_COUNT = 24.0
DEFAULT_RATING = "Sin clasificar"
DEFAULT_PUBLISHER = "Otros"  # No existe como columna dummy -> queda en 0 en todas (categoría base)


def construir_entrada(valores_formulario):
    """
    Construye un DataFrame de una sola fila, con las mismas columnas
    (y mismo orden) que el dataset usado para entrenar el modelo.

    valores_formulario: dict con claves 'plataforma', 'genero', 'anio', 'critic_score'
    """
    fila = {
        "Year_of_Release": float(valores_formulario.get("anio") or 2020),
        "Critic_Score": float(valores_formulario.get("critic_score") or 70),
        "Critic_Count": DEFAULT_CRITIC_COUNT,
        "User_Score": DEFAULT_USER_SCORE,
        "User_Count": DEFAULT_USER_COUNT,
    }

    df_entrada = pd.DataFrame([fila])

    # Variables categóricas a codificar igual que en el entrenamiento
    plataforma = valores_formulario.get("plataforma")
    genero = valores_formulario.get("genero")

    df_entrada["Platform"] = plataforma
    df_entrada["Genre"] = genero
    df_entrada["Rating"] = DEFAULT_RATING
    df_entrada["Publisher"] = DEFAULT_PUBLISHER

    columnas_categoricas = ["Platform", "Genre", "Publisher", "Rating"]
    df_entrada = pd.get_dummies(df_entrada, columns=columnas_categoricas, dtype=int)
    # NOTA: aquí NO se usa drop_first=True. A diferencia del entrenamiento
    # (que codifica miles de filas con muchas categorías por columna),
    # esta función siempre recibe una sola fila, por lo que cada columna
    # categórica tiene una única categoría. drop_first eliminaría esa
    # única categoría como "base", perdiendo por completo la selección
    # del usuario (todas las columnas dummy quedarían en 0 sin importar
    # la opción elegida). El reindex() de abajo ya se encarga de alinear
    # correctamente con las columnas reales del modelo entrenado.

    # Reindexar para que tenga EXACTAMENTE las mismas columnas que el
    # entrenamiento, en el mismo orden. Cualquier columna dummy que no
    # se haya generado (por ejemplo, una plataforma específica) se
    # completa con 0.
    df_entrada = df_entrada.reindex(columns=columnas_modelo, fill_value=0)

    return df_entrada


def predecir_ventas(valores_formulario):
    """
    Genera la predicción de ventas globales (en millones de copias)
    usando el modelo real entrenado.
    """
    entrada = construir_entrada(valores_formulario)
    prediccion = modelo.predict(entrada)
    return round(float(prediccion[0]), 2)


def obtener_estadisticas_dataset():
    """
    Estadísticas generales para la página principal.

    Se mantienen como valores fijos (calculados una vez sobre el
    dataset limpio) para no tener que cargar el CSV completo en cada
    request. Si se requiere mayor precisión, se puede reemplazar por
    una lectura real de data/ventas_videojuegos.csv con pandas.
    """
    return {
        "total_juegos": 16416,
        "plataformas": 31,
        "generos": 12,
        "anio_min": 1980,
        "anio_max": 2020,
    }