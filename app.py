import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de consistencia PEI", layout="wide")

st.title("Calculadora de Consistencia – Plan Estratégico Institucional UCCuyo 2023–2027")
st.markdown("Subí los archivos CSV con las actividades de uno o más objetivos específicos para ver su consistencia.")

uploaded_files = st.file_uploader(
    "Elegí los archivos CSV con las actividades (podés seleccionar varios)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            dfs.append(df)
            st.success(f"Archivo '{uploaded_file.name}' cargado correctamente.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error al cargar '{uploaded_file.name}': {e}")

    if dfs:
        df_actividades = pd.concat(dfs, ignore_index=True)

        columnas_requeridas = [
            'AÑO',
            'Objetivos específicos 1',
            'Acciones',
            'Responsable',
            'Unidad académica',
            'Indicadores'
        ]

        columnas_faltantes = [col for col in columnas_requeridas if col not in df_actividades.columns]

        if columnas_faltantes:
            st.error(f"Faltan las siguientes columnas requeridas: {', '.join(columnas_faltantes)}")
        else:
            st.success("Todos los encabezados requeridos están presentes.")
            st.subheader("Vista previa combinada de actividades")
            st.dataframe(df_actividades)

            resumen = df_actividades.groupby(['Unidad académica', 'Objetivos específicos 1']).size().reset_index(name='Cantidad de actividades')
            st.subheader("Resumen de actividades por unidad académica y objetivo específico")
            st.dataframe(resumen)
