import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="wide")

st.title("Calculadora de Consistencia PEI – UCCuyo 2023–2027")

st.markdown("Subí un archivo con las actividades institucionales para verificar su consistencia con el PEI.")

uploaded_file = st.file_uploader("Elegí el archivo CSV con las actividades", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("Archivo cargado correctamente.")
        st.dataframe(df.head())

        columnas_requeridas = [
            'AÑO',
            'Objetivos específicos 1',
            'Actividades Objetivo 1',
            'Detalle de la Actividad Objetivo 1'
        ]

        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if columnas_faltantes:
            st.error(f"Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
        else:
            for col in columnas_requeridas:
                df[col] = df[col].astype(str).str.strip().str.lower()

            st.success("Las columnas requeridas están presentes. Podés continuar con el análisis.")
            # Aquí se puede agregar el análisis de consistencia, como antes

    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {str(e)}")
else:
    st.info("Esperando que subas un archivo con las columnas esperadas.")
