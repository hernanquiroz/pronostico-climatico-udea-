import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_oni_data, load_medellin_data
from utils.enso_model import calibrate_and_predict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Pronóstico Climático Multinivel", layout="wide", page_icon="🌍")
st.title("🌍 Sistema de Pronóstico Climático: Global → Local")
st.markdown("Basado en el modelo *Delayed-Oscillator* (Kaper & Engler, Cap. 16) y datos históricos de IDEAM/SIATA.")

# ==========================================
# BARRA LATERAL (CONTROLES)
# ==========================================
st.sidebar.header("⚙️ Configuración del Modelo")
tau = st.sidebar.slider("Retardo de ondas (τ en meses)", 4, 10, 7, help="Tiempo que tardan las ondas de Rossby/Kelvin en cruzar el Pacífico.")

# ==========================================
# CARGA DE DATOS
# ==========================================
try:
    df_oni = load_oni_data()
    df_med = load_medellin_data()
    st.success("✅ Datos globales (ONI) y locales (Medellín) cargados correctamente desde la carpeta 'data/'.")
except FileNotFoundError as e:
    st.error(f"❌ Error de archivos: {e}")
    st.info("💡 Por favor, asegúrate de que los archivos `oni_real_cleaned.csv` y `medellin_clima_limpio.csv` estén en la carpeta `data/`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado al cargar datos: {e}")
    st.stop()

# ==========================================
# PROCESAMIENTO Y MODELO ENSO
# ==========================================
oni_history = df_oni['ONI'].values
model_results = calibrate_and_predict(oni_history, tau_months=tau, forecast_steps=12)

# Preparar fechas para el gráfico
last_date = df_oni['Date'].max()
future_dates = pd.date_range(start=last_date, periods=13, freq='ME')

df_plot_oni = pd.DataFrame({
    'Fecha': df_oni['Date'],
    'ONI_Real': oni_history
})

df_forecast = pd.DataFrame({
    'Fecha': future_dates,
    'ONI_Pronostico': [model_results['last_historical']] + list(model_results['forecast'])
})

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌊 1. Predicción Global (ONI)")
    
    fig_oni = go.Figure()
    fig_oni.add_trace(go.Scatter(x=df_plot_oni['Fecha'], y=df_plot_oni['ONI_Real'], mode='lines', name='ONI Histórico', line=dict(color='#1f77b4')))
    fig_oni.add_trace(go.Scatter(x=df_forecast['Fecha'], y=df_forecast['ONI_Pronostico'], mode='lines+markers', name=f'Pronóstico (τ={tau} meses)', line=dict(color='#d62728', dash='dash')))
    
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="Umbral El Niño (+0.5°C)")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="cyan", annotation_text="Umbral La Niña (-0.5°C)")
    
    fig_oni.update_layout(
        title="Histórico ONI y Proyección a 12 meses",
        xaxis_title="Fecha",
        yaxis_title="Anomalía de TSM en Niño 3.4 (°C)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_oni, use_container_width=True)
    
    st.caption(f"Parámetros calibrados del modelo: a={model_results['params']['a']:.3f}, b={model_results['params']['b']:.3f}, ε={model_results['params']['epsilon']:.4f}")

with col2:
    st.subheader("🌧️ 2. Impacto Local (Medellín)")
    
    metrica = st.selectbox("Selecciona la variable local a analizar:", ["temperatura_media_c", "precipitacion_mm"])
    
    # Agrupar por mes para ver estacionalidad (Cap. 9: Análisis de series de tiempo)
    df_monthly = df_med.groupby('mes_nombre')[metrica].mean().reset_index()
    
    # Ordenar meses cronológicamente
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_monthly['mes_nombre'] = pd.Categorical(df_monthly['mes_nombre'], categories=month_order, ordered=True)
    df_monthly = df_monthly.sort_values('mes_nombre')
    
    fig_local = px.bar(
        df_monthly, 
        x='mes_nombre', 
        y=metrica, 
        title=f"Promedio Histórico Mensual: {metrica.replace('_', ' ').title()}",
        color=metrica,
        color_continuous_scale='Blues' if metrica == 'precipitacion_mm' else 'Reds'
    )
    fig_local.update_layout(xaxis_title="Mes", yaxis_title="Valor Promedio")
    st.plotly_chart(fig_local, use_container_width=True)

# ==========================================
# MÓDULO DE CONSEJERÍA IA (Tu valor agregado)
# ==========================================
st.markdown("---")
st.subheader("🧠 Consejería Técnica Automatizada")

ultimo_oni = model_results['last_historical']
pronostico_oni_12m = model_results['forecast'][-1]

if ultimo_oni > 0.5 and pronostico_oni_12m < 0:
    st.warning(f"⚠️ **Alerta de Transición:** El modelo indica un El Niño actual (ONI: {ultimo_oni:.2f}°C) que transicionará hacia fase Neutra o La Niña en 12 meses (ONI proyectado: {pronostico_oni_12m:.2f}°C).")
    st.info("📌 **Recomendación para Medellín/Apartadó:** Históricamente (Cap. 9, Regresión y Cap. 19, Eventos Extremos), esta transición se correlaciona con un **aumento en la frecuencia e intensidad de lluvias**. Se recomienda revisar sistemas de drenaje, activar protocolos de monitoreo de quebradas (SIATA) y preparar planes de contingencia para movimientos en masa en laderas.")
    
elif ultimo_oni > 0.5 and pronostico_oni_12m > 0.5:
    st.warning(f"🔥 **Alerta de Persistencia:** Se proyecta que el El Niño (ONI actual: {ultimo_oni:.2f}°C) persista o se intensifique (ONI proyectado: {pronostico_oni_12m:.2f}°C).")
    st.info("📌 **Recomendación para Medellín/Apartadó:** Alto riesgo de déficit hídrico, islas de calor urbano y estrés en la red eléctrica. Priorizar planes de racionamiento de agua, monitoreo de temperatura superficial (Cap. 12, Zonal EBM) y protección de cultivos sensibles a la sequía.")
    
elif ultimo_oni < -0.5 and pronostico_oni_12m < -0.5:
    st.info(f"❄️ **Fase La Niña Persistente:** ONI actual: {ultimo_oni:.2f}°C, proyección: {pronostico_oni_12m:.2f}°C.")
    st.info("📌 **Recomendación:** Condiciones típicas de mayor nubosidad y precipitación en el Valle de Aburrá. Monitorear niveles de quebradas y prevenir inundaciones locales, aunque las temperaturas máximas suelen ser ligeramente más bajas.")
    
else:
    st.success(f"✅ **Condiciones Normales:** El ONI actual es {ultimo_oni:.2f}°C con una proyección estable de {pronostico_oni_12m:.2f}°C. Mantener monitoreo rutinario y seguir los promedios históricos estacionales mostrados en el gráfico local.")
