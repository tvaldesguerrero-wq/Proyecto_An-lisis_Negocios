# Análisis y predicción de ventas de videojuegos

Proyecto semestral — Análisis de Negocios (2026-01) — Equipo ADN

## Objetivo de la predicción

Estimar las ventas globales (`Global_Sales`, en millones de copias) de un
videojuego a partir de variables como plataforma, género, año de lanzamiento
y puntaje de la crítica especializada.

## Modelo utilizado

> **Pendiente de completar por el equipo de modelamiento.**
> Indicar aquí: tipo de modelo final elegido (ej. Random Forest Regressor,
> Regresión Lineal, etc.), métricas obtenidas (RMSE, R², etc.) y por qué se
> seleccionó por sobre los otros modelos candidatos.

## Estructura del proyecto

```
proyecto-videojuegos-web/
├── app.py                  # Servidor Flask principal
├── requirements.txt        # Librerías necesarias
├── static/
│   ├── css/style.css        # Estilos de la app
│   └── img/                 # Imágenes (si se requieren)
├── templates/
│   ├── base.html            # Plantilla base (Jinja)
│   ├── index.html           # Página principal
│   └── prediccion.html      # Página de predicciones
├── models/
│   └── modelo_ventas.pkl    # Modelo entrenado (pendiente)
├── utils/
│   └── model_utils.py       # Carga del modelo y lógica de predicción
└── data/
    └── ventas_videojuegos.csv   # Dataset utilizado
```

## Cómo ejecutar la aplicación localmente

1. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1      # Windows (PowerShell)
   source venv/bin/activate       # Mac/Linux
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el servidor:
   ```bash
   python app.py
   ```

4. Abrir en el navegador:
   ```
   http://127.0.0.1:5000
   ```

## Navegación de la aplicación

- **Inicio (`/`)**: presenta estadísticas generales del dataset utilizado.
- **Predicción (`/prediccion`)**: formulario donde se ingresan las
  características de un videojuego (plataforma, género, año, puntaje de
  críticos) y se muestra la estimación de ventas globales generada por el
  modelo.

## Estado actual del desarrollo

Este scaffold usa una función de predicción **simulada** en
`utils/model_utils.py`, de modo que la aplicación es completamente funcional
de extremo a extremo mientras se entrena el modelo definitivo. Una vez
disponible el archivo `.pkl` real, basta con:

1. Colocarlo en `models/`.
2. Actualizar `utils/model_utils.py` para cargarlo con `pickle` o `joblib`
   y usarlo en lugar de la lógica simulada (instrucciones detalladas como
   comentarios dentro de ese mismo archivo).

No es necesario modificar `app.py` ni los templates HTML para esta integración.
