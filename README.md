# FraudShield AI 🛡️
FraudShield AI es un sistema inteligente de detección de fraudes en transacciones de comercio electrónico. Cuenta con un dashboard de monitoreo en tiempo real construido con **Streamlit** y una API de inferencia rápida y robusta construida con **FastAPI**. El motor de detección está impulsado por un modelo de Machine Learning basado en **XGBoost**.
## 🚀 Características Principales

FraudShield AI es un sistema inteligente de detección de fraudes en transacciones de comercio electrónico. Cuenta con un dashboard de monitoreo en tiempo real construido con **Streamlit** y una API de inferencia rápida y robusta construida con **FastAPI**. El motor de detección está impulsado por un modelo de Machine Learning basado en **XGBoost**.

## 🚀 Características Principales

*   **Detección en Tiempo Real:** API RESTful que recibe datos de transacciones y devuelve la probabilidad de fraude y el nivel de riesgo instantáneamente.
*   **Pipeline de Machine Learning:** Preprocesamiento completo que incluye Label Encoding, Standard Scaling y Feature Selection (SelectFromModel), entrenado con balanceo de clases usando SMOTE.
*   **Dashboard Interactivo:** Un panel de control moderno para visualizar métricas clave (Accuracy, Precision, Recall, F1-Score, ROC-AUC), tendencias recientes, fraudes por dispositivo y alertas en tiempo real.
*   **Análisis Exploratorio:** Pestañas dedicadas para explorar distribuciones demográficas, mapas de calor, transacciones sospechosas y la salud del modelo en producción.
## 📁 Estructura del Proyecto

## 📁 Estructura del Proyecto

```text
MLOps/
├── api/
│   └── main.py                 # Backend (FastAPI) para predicción de modelos
├── app/
│   ├── assets/                 # Estilos CSS personalizados (style.css)
│   ├── components/             # Componentes reusables de UI de Streamlit (sidebar)
│   ├── data/                   # Datasets y scripts de carga de datos (data_loader.py)
│   ├── models/                 # Modelos entrenados (.pkl) y metadatos (model_info.json)
│   ├── pages/                  # Vistas secundarias de Streamlit (Análisis, Alertas, Predicciones, etc.)
│   └── streamlit_app.py        # Archivo principal de la aplicación de frontend
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación del proyecto
```
## 🛠️ Tecnologías Utilizadas
*   **Backend:** FastAPI, Pydantic, Uvicorn
*   **Frontend:** Streamlit, Plotly
*   **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Numpy
## ⚙️ Instalación y Ejecución
Sigue estos pasos para correr el proyecto localmente.
### 1. Clonar el Repositorio e Instalar Dependencias

## 🛠️ Tecnologías Utilizadas

*   **Backend:** FastAPI, Pydantic, Uvicorn
*   **Frontend:** Streamlit, Plotly
*   **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Numpy

## ⚙️ Instalación y Ejecución

Sigue estos pasos para correr el proyecto localmente.

### 1. Clonar el Repositorio e Instalar Dependencias

```bash
# Clonar el proyecto (Reemplaza con tu URL real)
git clone https://github.com/TuUsuario/fraudshield-ai.git
cd fraudshield-ai
# Instalar los requerimientos
pip install -r requirements.txt
```
### 2. Ejecutar la API (FastAPI)
En una terminal, levanta el servidor backend para que el modelo esté disponible:

# Instalar los requerimientos
pip install -r requirements.txt
```

### 2. Ejecutar la API (FastAPI)

En una terminal, levanta el servidor backend para que el modelo esté disponible:

```bash
# Iniciar servidor Uvicorn en el puerto 8000
python -m uvicorn api.main:app --reload
```
*La API estará disponible en `http://localhost:8000`. Puedes ver la documentación interactiva en `http://localhost:8000/docs`.*
### 3. Ejecutar el Dashboard (Streamlit)
Abre una **segunda terminal** y ejecuta el dashboard interactivo:

### 3. Ejecutar el Dashboard (Streamlit)

Abre una **segunda terminal** y ejecuta el dashboard interactivo:

```bash
# Iniciar aplicación de Streamlit en el puerto 8501
streamlit run app/streamlit_app.py
```
*El Dashboard abrirá automáticamente en tu navegador en `http://localhost:8501`.*

---
*Desarrollado como proyecto de Machine Learning / MLOps.*
