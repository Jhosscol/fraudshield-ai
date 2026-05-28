import streamlit as st
from datetime import datetime

def render_sidebar():
    """Renders the common sidebar for all pages."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #58a6ff; font-size: 24px; margin-bottom: 0;">🛡️ FraudShield AI</h1>
                <p style="color: #8b949e; font-size: 12px; margin-top: 0;">Intelligent Detection System</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # System Status Card
        st.markdown(
            """
            <div style="background-color: rgba(46, 160, 67, 0.1); border: 1px solid #2ea043; border-radius: 8px; padding: 10px; margin-bottom: 20px;">
                <h4 style="color: #3fb950; margin: 0; font-size: 14px;">🟢 Estado del Sistema</h4>
                <p style="color: #c9d1d9; margin: 5px 0 0 0; font-size: 12px;">Operando con normalidad</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Model Info
        st.markdown("### 🤖 Modelo de IA")
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #8b949e;">
                <p style="margin: 2px 0;"><b>Versión:</b> XGBoost v2.5.0</p>
                <p style="margin: 2px 0;"><b>Último Entto:</b> 25-May-2026</p>
                <p style="margin: 2px 0;"><b>Latencia API:</b> ~45ms</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        st.info("💡 Selecciona una sección arriba para navegar.")
