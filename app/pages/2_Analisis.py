import streamlit as st
import plotly.express as px
from pathlib import Path
from data.data_loader import load_real_transactions
from components.sidebar import render_sidebar

st.set_page_config(page_title="Análisis de Fraude", page_icon="📈", layout="wide")

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

st.title("📈 Análisis de Fraude")
st.markdown("Exploración profunda de patrones y distribuciones de fraude.")

# Filters
st.subheader("Filtros de Datos")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    locs = ['Todos'] + list(df['Customer Location'].unique())
    selected_loc = st.selectbox("País / Ubicación", locs)

with col_f2:
    payments = ['Todos'] + list(df['Payment Method'].unique())
    selected_payment = st.selectbox("Método de Pago", payments)

with col_f3:
    devices = ['Todos'] + list(df['Device Used'].unique())
    selected_device = st.selectbox("Dispositivo", devices)

with col_f4:
    hours = ['Todas'] + sorted(list(df['Transaction Hour'].unique()))
    selected_hour = st.selectbox("Hora de Transacción", hours)

# Apply filters
filtered_df = df.copy()
if selected_loc != 'Todos':
    filtered_df = filtered_df[filtered_df['Customer Location'] == selected_loc]
if selected_payment != 'Todos':
    filtered_df = filtered_df[filtered_df['Payment Method'] == selected_payment]
if selected_device != 'Todos':
    filtered_df = filtered_df[filtered_df['Device Used'] == selected_device]
if selected_hour != 'Todas':
    filtered_df = filtered_df[filtered_df['Transaction Hour'] == int(selected_hour)]

fraud_df = filtered_df[filtered_df['Is Fraud'] == 1]

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Fraudes por País")
    if not fraud_df.empty:
        loc_counts = fraud_df['Customer Location'].value_counts().reset_index()
        loc_counts.columns = ['País', 'Cantidad']
        fig_loc = px.bar(loc_counts, x='País', y='Cantidad',
                         template='plotly_dark',
                         color='Cantidad',
                         color_continuous_scale='Reds')
        fig_loc.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_loc, use_container_width=True)
    else:
        st.info("No hay datos de fraude para estos filtros.")

with col2:
    st.markdown("#### Fraudes por Dispositivo")
    if not fraud_df.empty:
        device_counts = fraud_df['Device Used'].value_counts().reset_index()
        device_counts.columns = ['Dispositivo', 'Cantidad']
        fig_dev = px.pie(device_counts, values='Cantidad', names='Dispositivo',
                         template='plotly_dark',
                         hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_dev.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dev, use_container_width=True)
    else:
        st.info("No hay datos de fraude para estos filtros.")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Fraudes por Hora")
    if not fraud_df.empty:
        hour_counts = fraud_df['Transaction Hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['Hora', 'Cantidad']
        fig_hour = px.line(hour_counts, x='Hora', y='Cantidad',
                           template='plotly_dark',
                           markers=True,
                           line_shape='spline',
                           color_discrete_sequence=['#ff7b72'])
        fig_hour.update_layout(xaxis=dict(tickmode='linear', dtick=2), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No hay datos de fraude para estos filtros.")

with col4:
    st.markdown("#### Distribución de Montos (Todos vs Fraude)")
    fig_amount = px.box(filtered_df, x='Is Fraud', y='Transaction Amount',
                        template='plotly_dark',
                        color='Is Fraud',
                        color_discrete_map={0: '#58a6ff', 1: '#ff7b72'})
    fig_amount.update_layout(xaxis_title="Es Fraude (0=No, 1=Sí)", yaxis_title="Monto ($)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_amount, use_container_width=True)
