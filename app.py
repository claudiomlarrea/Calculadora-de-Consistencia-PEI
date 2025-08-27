
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="wide")

st.title("Calculadora de Consistencia con el PEI")
st.markdown("Subí un archivo con tus actividades institucionales y comparalo con los objetivos, acciones e indicadores del PEI.")

uploaded_file = st.file_uploader("Subí tu archivo de actividades", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_actividades = pd.read_csv(uploaded_file)
        else:
            df_actividades = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
    else:
        st.success("Archivo cargado correctamente. Mostrando vista previa:")
        st.dataframe(df_actividades.head())

        # Cargar la base de referencia
        df_referencia = pd.read_csv("pei_referencia.csv")

        # Normalización de columnas
        for col in ["Objetivo", "Acción", "Indicador"]:
            df_actividades[col] = df_actividades[col].astype(str).str.strip().str.lower()
            df_referencia[col] = df_referencia[col].astype(str).str.strip().str.lower()

        # Comparación
        df_resultado = df_actividades.copy()
        df_resultado["Coincide Objetivo"] = df_resultado["Objetivo"].isin(df_referencia["Objetivo"])
        df_resultado["Coincide Acción"] = df_resultado["Acción"].isin(df_referencia["Acción"])
        df_resultado["Coincide Indicador"] = df_resultado["Indicador"].isin(df_referencia["Indicador"])

        st.markdown("### Resultados del análisis")
        st.dataframe(df_resultado)

        # Descargar resultado
        csv = df_resultado.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar resultados en CSV", data=csv, file_name="resultado_consistencia.csv", mime="text/csv")
