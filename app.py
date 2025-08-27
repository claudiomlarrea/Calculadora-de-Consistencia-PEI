import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de consistencia PEI", layout="wide")

st.title("Calculadora de Consistencia - Plan Estratégico Institucional UCCuyo 2023–2027")

st.markdown("Subí un archivo con las actividades de un objetivo específico para ver su consistencia.")

uploaded_file = st.file_uploader("Elegí el archivo CSV con las actividades", type="csv")

if uploaded_file is not None:
    try:
        df_actividades = pd.read_csv(uploaded_file)

        st.success("Archivo cargado correctamente. Mostrando vista previa:")
        st.dataframe(df_actividades.head())

        # Columnas esperadas
        columnas_requeridas = [
            'AÑO',
            'Objetivos específicos 1',
            'Actividades Objetivo 1',
            'Detalle de la Actividad Objetivo 1'
        ]

        # Verificar si están todas las columnas necesarias
        faltantes = [col for col in columnas_requeridas if col not in df_actividades.columns]
        if faltantes:
            st.error(f"Las siguientes columnas faltan en el archivo: {', '.join(faltantes)}")
        else:
            # Limpiar texto de columnas clave
            for col in columnas_requeridas:
                df_actividades[col] = df_actividades[col].astype(str).str.strip().str.lower()

            # Acá podrías seguir con más análisis (consistencia, reportes, etc.)
            st.success("Las columnas son válidas. Podés continuar con el análisis.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")
else:
    st.info("Esperando que subas un archivo CSV válido.")
