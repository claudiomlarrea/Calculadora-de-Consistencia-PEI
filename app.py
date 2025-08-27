
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="wide")
st.title("🧮 Calculadora de Consistencia entre Actividades y Objetivos del PEI")

st.markdown("### 1. Subí el archivo con las actividades")
uploaded_file = st.file_uploader("Archivo Excel con actividades (formato .xlsx)", type=["xlsx"])

if uploaded_file:
    actividades = pd.read_excel(uploaded_file)
    st.success("✅ Archivo cargado correctamente")
    st.write("Vista previa de las actividades:", actividades.head())

    st.markdown("### 2. Objetivos institucionales de referencia")
    try:
        referencia = pd.read_csv("pei_referencia.csv")
        st.write(referencia)
    except Exception as e:
        st.error(f"Error al leer pei_referencia.csv: {e}")

    st.markdown("### 3. Procesamiento de consistencia")
    if "Objetivo PEI" in actividades.columns and "Descripción Actividad" in actividades.columns:
        actividades["Consistente"] = actividades["Objetivo PEI"].isin(referencia["Objetivo"])
        resumen = actividades["Consistente"].value_counts().rename(index={True: "Consistentes", False: "Inconsistentes"})
        st.dataframe(actividades)
        st.markdown("#### Resumen")
        st.write(resumen)
    else:
        st.warning("⚠️ Las columnas necesarias no están presentes en el archivo cargado.")
