import pandas as pd
import os

def load_oni_data(file_path="data/oni_real_cleaned.csv"):
    """Carga y valida los datos del índice ONI."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}. Asegúrate de que esté en la carpeta 'data/'.")
    
    df = pd.read_csv(file_path)
    # Asegurar nombres de columna estándar
    expected_cols = ['Year', 'Period', 'ONI']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El archivo ONI debe tener las columnas: {expected_cols}")
    
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Period'].map(
        {'DJF':'02', 'JFM':'03', 'FMA':'04', 'MAM':'05', 'AMJ':'06', 'MJJ':'07', 
         'JJA':'08', 'JAS':'09', 'ASO':'10', 'SON':'11', 'OND':'12', 'NDJ':'01'}
    ), errors='coerce')
    
    return df.dropna(subset=['Date']).sort_values('Date')

def load_medellin_data(file_path="data/medellin_clima_limpio.csv"):
    """Carga y valida los datos climáticos de Medellín."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}. Asegúrate de que esté en la carpeta 'data/'.")
    
    df = pd.read_csv(file_path)
    expected_cols = ['fecha', 'temperatura_media_c', 'precipitacion_mm']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El archivo de Medellín debe tener las columnas: {expected_cols}")
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.dropna(subset=['fecha', 'temperatura_media_c', 'precipitacion_mm'])
    df = df.sort_values('fecha')
    
    # Agregar columna de mes para análisis estacional
    df['mes'] = df['fecha'].dt.month
    df['mes_nombre'] = df['fecha'].dt.month_name(locale='es_ES') # Requiere configuración de locale, fallback a inglés si falla
    
    return df
