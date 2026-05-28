import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path
from data.data_loader import get_model_metrics, get_confusion_matrix_data
from components.sidebar import render_sidebar

st.set_page_config(page_title="Monitoreo del Modelo", page_icon="🤖", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

render_sidebar()

st.title("🤖 Monitoreo de Rendimiento del Modelo")
st.markdown("Supervisa la salud, precisión y degradación del modelo de detección de fraude en producción.")

metrics = get_model_metrics()

# Metrics Overview
st.subheader("Métricas Actuales (XGBoost v2.5.0)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Exactitud (Accuracy)", f"{metrics['Accuracy']:.3f}")
with col2:
    st.metric("Precisión (Precision)", f"{metrics['Precision']:.3f}")
with col3:
    st.metric("Exhaustividad (Recall)", f"{metrics['Recall']:.3f}")
with col4:
    st.metric("Puntuación F1", f"{metrics['F1 Score']:.3f}")
with col5:
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

st.divider()

col_cm, col_drift = st.columns(2)

with col_cm:
    st.subheader("Matriz de Confusión")
    st.markdown("*(Últimas 10,000 predicciones)*")
    z = get_confusion_matrix_data()
    x = ['Legítimo Predicho', 'Fraude Predicho']
    y = ['Legítimo Real', 'Fraude Real']
    
    # Reverse y for standard orientation where Actual Legitimate is on top
    z_rev = z[::-1]
    y_rev = y[::-1]
    
    fig_cm = ff.create_annotated_heatmap(
        z_rev, x=x, y=y_rev, 
        colorscale='Blues',
        showscale=True
    )
    # Update layout to fit dark theme
    fig_cm.update_layout(template='plotly_dark', margin=dict(t=30, l=150, r=0, b=0))
    st.plotly_chart(fig_cm, use_container_width=True)

with col_drift:
    st.subheader("Curva ROC")
    # Simulate an ROC curve
    import numpy as np
    import pandas as pd
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr)**3 # fake curve
    roc_df = pd.DataFrame({'Tasa Falsos Positivos': fpr, 'Tasa Verdaderos Positivos': tpr})
    
    fig_roc = px.line(roc_df, x='Tasa Falsos Positivos', y='Tasa Verdaderos Positivos', 
                      title=f'Curva ROC (AUC = {metrics["ROC-AUC"]:.3f})',
                      template='plotly_dark')
    fig_roc.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_roc.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(fig_roc, use_container_width=True)

st.info("💡 Nota: El monitoreo de data drift (deriva de datos) y la importancia de características están actualmente deshabilitados en este entorno.")
