import streamlit as st
import time
import random
from pathlib import Path
from datetime import datetime
from components.sidebar import render_sidebar

st.set_page_config(page_title="Predicción Manual", page_icon="🔍", layout="centered")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

render_sidebar()

st.title("🔍 Predicción Manual de Transacciones")
st.markdown("Ingresa los detalles de la transacción para ejecutar una inferencia a través del modelo de FraudShield AI.")

# Define the model feature schema grouped by sections
MODEL_FEATURES = {
    "Información de la Transacción": {
        "Transaction Amount": {"label": "Monto de la Transacción ($)", "type": "number", "min": 0.0, "value": 150.0, "step": 10.0},
        "Payment Method": {"label": "Método de Pago", "type": "categorical", "options": ['Credit Card', 'Debit Card', 'PayPal', 'Crypto', 'Bank Transfer']},
        "Product Category": {"label": "Categoría de Producto", "type": "categorical", "options": ['Electronics', 'Clothing', 'Home & Garden', 'Digital Goods', 'Toys']},
        "Quantity": {"label": "Cantidad de Artículos", "type": "number", "min": 1, "value": 1, "step": 1, "is_int": True},
        "Transaction Hour": {"label": "Hora de la Transacción (0-23)", "type": "number", "min": 0, "max": 23, "value": datetime.now().hour, "is_int": True}
    },
    "Información del Cliente": {
        "Customer ID": {"label": "ID del Cliente", "type": "text", "value": "CUS-1029"},
        "Customer Location": {"label": "Ubicación del Cliente", "type": "categorical", "options": ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'IN', 'BR', 'MX', 'JP']},
        "Customer Age": {"label": "Edad del Cliente", "type": "number", "min": 18, "max": 100, "value": 35, "is_int": True},
        "Account Age Days": {"label": "Antigüedad de la Cuenta (Días)", "type": "number", "min": 0, "value": 300, "step": 10, "is_int": True}
    },
    "Información del Dispositivo": {
        "Device Used": {"label": "Dispositivo Utilizado", "type": "categorical", "options": ['Mobile', 'Desktop', 'Tablet']}
    }
}

def render_dynamic_form(schema):
    input_data = {}
    
    for section_name, features in schema.items():
        st.markdown(f"#### {section_name}")
        
        # Determine number of columns based on features (max 2 for better layout)
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        
        for i, (feat_name, config) in enumerate(features.items()):
            current_col = cols[i % 2]
            with current_col:
                label = config.get("label", feat_name)
                if config["type"] == "number":
                    if config.get("is_int"):
                        input_data[feat_name] = st.number_input(
                            label, 
                            min_value=int(config.get("min", 0)), 
                            max_value=int(config.get("max", 100000)) if "max" in config else None,
                            value=int(config.get("value", 0)), 
                            step=int(config.get("step", 1)),
                            key=feat_name
                        )
                    else:
                        input_data[feat_name] = st.number_input(
                            label, 
                            min_value=float(config.get("min", 0.0)), 
                            max_value=float(config.get("max", 100000.0)) if "max" in config else None,
                            value=float(config.get("value", 0.0)), 
                            step=float(config.get("step", 1.0)),
                            key=feat_name
                        )
                elif config["type"] == "categorical":
                    input_data[feat_name] = st.selectbox(
                        label, 
                        options=config.get("options", []),
                        key=feat_name
                    )
                elif config["type"] == "text":
                    input_data[feat_name] = st.text_input(
                        label, 
                        value=config.get("value", ""),
                        key=feat_name
                    )
        st.markdown("<hr style='margin: 1rem 0; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
    return input_data

with st.form("prediction_form"):
    input_data = render_dynamic_form(MODEL_FEATURES)
        
    submit_button = st.form_submit_button("Analizar Transacción", use_container_width=True)

if submit_button:
    with st.spinner("Consultando API de FraudShield AI..."):
        import requests
        
        api_url = "http://localhost:8000/predict"
        payload = {
            "transaction_amount": float(input_data.get("Transaction Amount", 0.0)),
            "customer_id": str(input_data.get("Customer ID", "CUS-0000")),
            "payment_method": str(input_data.get("Payment Method", "")),
            "product_category": str(input_data.get("Product Category", "")),
            "quantity": int(input_data.get("Quantity", 1)),
            "customer_age": int(input_data.get("Customer Age", 35)),
            "customer_location": str(input_data.get("Customer Location", "")),
            "device_used": str(input_data.get("Device Used", "")),
            "account_age_days": int(input_data.get("Account Age Days", 300)),
            "transaction_hour": int(input_data.get("Transaction Hour", 12))
        }
        
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                final_prob = result["fraud_probability"]
                risk_level = result["risk_level"]
                
                st.subheader("Resultado de la Predicción")
                st.write(f"**Probabilidad de Fraude:** {final_prob*100:.1f}%")
                st.progress(min(final_prob, 1.0))
                
                if risk_level == "HIGH":
                    st.error("🚨 RIESGO ALTO: La transacción ha sido marcada como potencialmente fraudulenta.")
                elif risk_level == "MEDIUM":
                    st.warning("⚠️ RIESGO MEDIO: La transacción requiere revisión manual.")
                else:
                    st.success("✅ RIESGO BAJO: La transacción parece legítima.")
                    
                with st.expander("Explicación del Modelo (Features Enviados)"):
                    st.write(payload)
            else:
                st.error(f"Error en la API: {response.text}")
        except Exception as e:
            st.error(f"No se pudo conectar a la API. Asegúrate de que FastAPI esté corriendo en el puerto 8000. Error: {e}")

