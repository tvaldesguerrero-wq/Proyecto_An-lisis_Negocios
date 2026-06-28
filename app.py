"""
app.py
Archivo principal que inicia el servidor Flask.
"""

from flask import Flask, render_template, request
from utils.model_utils import (
    predecir_ventas,
    obtener_estadisticas_dataset,
    obtener_feature_importance,
)

app = Flask(__name__)


@app.route("/")
def home():
    stats = obtener_estadisticas_dataset()
    importancia = obtener_feature_importance(top_n=6)
    return render_template("index.html", stats=stats, importancia=importancia)


@app.route("/prediccion", methods=["GET", "POST"])
def prediccion():
    resultado = None
    valores_formulario = {}
    importancia = obtener_feature_importance(top_n=8)

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
        importancia=importancia,
    )


if __name__ == "__main__":
    app.run(debug=True)
