"""
app.py
Archivo principal que inicia el servidor Flask.

NOTA PARA EL EQUIPO:
Este scaffold usa datos DUMMY (de prueba) generados en utils/model_utils.py.
Cuando el modelo real (.pkl) esté listo, solo hay que:
  1.venv\Scripts\Activate.ps1
  2. Reemplazar las funciones dummy en utils/model_utils.py por la carga real del modelo
  3. No es necesario tocar este archivo ni los templates HTML.
"""

from flask import Flask, render_template, request
from utils.model_utils import predecir_ventas, obtener_estadisticas_dataset

app = Flask(__name__)


@app.route("/")
def home():
    """Página principal: presentación del proyecto y estadísticas generales."""
    stats = obtener_estadisticas_dataset()
    return render_template("index.html", stats=stats)


@app.route("/prediccion", methods=["GET", "POST"])
def prediccion():
    """Página de predicciones: formulario + resultado del modelo."""
    resultado = None
    valores_formulario = {}

    if request.method == "POST":
        valores_formulario = {
            "plataforma": request.form.get("plataforma"),
            "genero": request.form.get("genero"),
            "anio": request.form.get("anio"),
            "critic_score": request.form.get("critic_score"),
        }
        resultado = predecir_ventas(valores_formulario)

    return render_template(
        "prediccion.html",
        resultado=resultado,
        valores=valores_formulario,
    )


if __name__ == "__main__":
    app.run(debug=True)
