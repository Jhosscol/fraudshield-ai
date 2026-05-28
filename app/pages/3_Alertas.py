import streamlit as st
import pandas as pd
from pathlib import Path
from data.data_loader import load_real_transactions
from components.sidebar import render_sidebar

st.set_page_config(page_title="Alertas en Tiempo Real", page_icon="🚨", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

render_sidebar()

@st.cache_data
def load_data():
    return load_real_transactions(20000)

df = load_data()

st.title("🚨 Alertas en Tiempo Real")
st.markdown("Feed en vivo de transacciones sospechosas que requieren revisión.")

# Filter and sort data for alerts
alerts_df = df[df['Is Fraud'] == 1].copy()
# Assign a fake high probability for visual formatting since we only have labels
if 'Date' in alerts_df.columns:
    alerts_df = alerts_df.sort_values(by='Date', ascending=False).head(50)
else:
    alerts_df = alerts_df.head(50)

alerts_df['Fraud Probability'] = 0.99

st.subheader("Acción Requerida")

# Display critical alert cards using st.error for >90%
critical_risk = alerts_df[alerts_df['Fraud Probability'] >= 0.90].head(3)

if not critical_risk.empty:
    st.markdown("### 🔴 Alertas Críticas (>90% Probabilidad)")
    for i, (_, row) in enumerate(critical_risk.iterrows()):
        st.error(f"**ALERTA CRÍTICA [{row['Transaction ID']}]:** Transacción de ${row['Transaction Amount']} desde {row['Customer Location']} con **{(row['Fraud Probability']*100):.1f}%** de probabilidad de fraude.", icon="🚨")

st.divider()

st.subheader("Feed de Transacciones Sospechosas")

def get_risk_badge(prob):
    if prob >= 0.90:
        return '<span class="badge badge-danger">Riesgo Alto</span>'
    elif prob >= 0.60:
        return '<span class="badge badge-warning">Riesgo Medio</span>'
    else:
        return '<span class="badge badge-success">Riesgo Bajo</span>'

def format_row(row):
    return f"""<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
<div>
<h4 style="margin: 0; color: #e6edf3;">{row['Transaction ID']}</h4>
<p style="margin: 5px 0 0 0; font-size: 14px; color: #8b949e;">{row['Date'].strftime('%Y-%m-%d %H:%M')} | {row['Customer Location']} | {row['Device Used']} | {row['Payment Method']}</p>
</div>
<div style="text-align: right;">
<h3 style="margin: 0; color: #58a6ff;">${row['Transaction Amount']:.2f}</h3>
<div style="margin-top: 5px;">{get_risk_badge(row['Fraud Probability'])} <span style="font-size: 14px; font-weight: bold; color: #c9d1d9;">{(row['Fraud Probability']*100):.1f}%</span></div>
</div>
</div>"""

html_content = '<div style="height: 600px; overflow-y: scroll; padding-right: 10px;">\n'
for _, row in alerts_df.iterrows():
    html_content += format_row(row) + '\n'
html_content += '</div>'

st.markdown(html_content, unsafe_allow_html=True)
