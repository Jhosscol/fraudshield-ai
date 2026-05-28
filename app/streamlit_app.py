import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from data.data_loader import load_real_transactions, get_model_metrics
from components.sidebar import render_sidebar

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="FraudShield AI - Panel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()
render_sidebar()

# Cache data loading
@st.cache_data
def load_data():
    return load_real_transactions(20000)

@st.cache_data
def load_metrics():
    return get_model_metrics()

df = load_data()
metrics = load_metrics()

# --- Main Dashboard ---
st.title("🛡️ FraudShield AI")
st.markdown("### Panel de Control de Transacciones")
st.markdown("Monitoreo en tiempo real para detección de fraudes en comercio electrónico.")

st.divider()

# KPI Metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_txns = len(df)
fraud_txns = df['Is Fraud'].sum()
fraud_percentage = (fraud_txns / total_txns) * 100 if total_txns > 0 else 0

with col1:
    st.metric(label="Total Transacciones", value=f"{total_txns:,}")

with col2:
    st.metric(label="Fraudes Detectados", value=f"{fraud_txns:,}", delta=f"{fraud_percentage:.1f}%", delta_color="inverse")

with col3:
    st.metric(label="Tasa de Fraude", value=f"{fraud_percentage:.2f}%")

with col4:
    st.metric(label="Precisión del Modelo", value=f"{metrics['Accuracy']*100:.1f}%", delta="Óptimo")

with col5:
    st.metric(label="Última Actualización", value="Justo ahora")

st.divider()

# Quick Overview Charts
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("Tendencia de Fraudes Recientes")
    # Group by date for a trend
    if 'Date' in df.columns:
        df['DateOnly'] = pd.to_datetime(df['Date']).dt.date
        trend_df = df.groupby('DateOnly')['Is Fraud'].sum().reset_index()
        fig = px.line(trend_df, x='DateOnly', y='Is Fraud', 
                      title='Transacciones Fraudulentas Diarias',
                      template='plotly_dark',
                      line_shape='spline',
                      color_discrete_sequence=['#ff7b72'])
        fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sin datos de fecha disponibles.")

with col_chart2:
    st.subheader("Fraudes por Dispositivo")
    if 'Device Used' in df.columns:
        fraud_devices = df[df['Is Fraud'] == 1]['Device Used'].value_counts().reset_index()
        fraud_devices.columns = ['Device Used', 'Count']
        fig2 = px.pie(fraud_devices, values='Count', names='Device Used', 
                      title='Dispositivos más usados en fraudes',
                      template='plotly_dark',
                      hole=0.4,
                      color_discrete_sequence=['#58a6ff', '#8b949e', '#1f6feb'])
        fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Sin datos de dispositivos disponibles.")

st.divider()

st.header("🔍 Predicción de Fraude en Tiempo Real")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input("Monto de Transacción", min_value=0.0, value=5000.0)
    customer_id = st.text_input("Customer ID", value="CUST100")
    payment_method = st.selectbox(
        "Método de Pago",
        ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
    )
    product_category = st.selectbox(
        "Categoría",
        ["Electronics", "Clothing", "Home", "Beauty"]
    )
    quantity = st.number_input("Cantidad", min_value=1, value=1)

with col2:
    customer_age = st.number_input("Edad Cliente", min_value=18, max_value=100, value=30)
    customer_location = st.text_input("Ubicación", value="New York")
    device_used = st.selectbox(
        "Dispositivo",
        ["Mobile", "Desktop", "Tablet"]
    )
    account_age_days = st.number_input("Edad de Cuenta (días)", min_value=0, value=365)
    transaction_hour = st.slider("Hora de Transacción", 0, 23, 21)

if st.button("🚨 Detectar Fraude"):

    payload = {
        "transaction_amount": transaction_amount,
        "customer_id": customer_id,
        "payment_method": payment_method,
        "product_category": product_category,
        "quantity": quantity,
        "customer_age": customer_age,
        "customer_location": customer_location,
        "device_used": device_used,
        "account_age_days": account_age_days,
        "transaction_hour": transaction_hour
    }

    try:
        response = requests.post(
            "https://fraudshield-ai-oc40.onrender.com/predict",
            json=payload
        )

        result = response.json()

        fraud_prob = result["fraud_probability"]
        risk_level = result["risk_level"]

        st.success(f"Probabilidad de Fraude: {fraud_prob:.4f}")

        if risk_level == "HIGH":
            st.error(f"Nivel de Riesgo: {risk_level}")

        elif risk_level == "MEDIUM":
            st.warning(f"Nivel de Riesgo: {risk_level}")

        else:
            st.info(f"Nivel de Riesgo: {risk_level}")

    except Exception as e:
        st.error(f"Error conectando con API: {e}")
