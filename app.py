
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="wide")

st.title("Calculadora de Consistencia de Actividades vs PEI")
st.markdown("Esta herramienta permite evaluar si las actividades institucionales están alineadas con el Plan Estratégico Institucional (PEI) UCCuyo 2023–2027.")

uploaded_file = st.file_uploader("Cargar archivo de actividades (Excel)", type=["xlsx"])
pei_file = "pei_referencia.csv"

if uploaded_file:
    actividades_df = pd.read_excel(uploaded_file)
    pei_df = pd.read_csv(pei_file)

    # Normalizar texto
    actividades_df.fillna("", inplace=True)
    pei_df.fillna("", inplace=True)

    resultados = []
    for _, act in actividades_df.iterrows():
        descripcion = act["Descripción de la Actividad"].lower()
        correspondencias = pei_df[pei_df["Acción"].str.lower().str.contains(descripcion, na=False)]
        if not correspondencias.empty:
            nivel = "Plena correspondencia"
            ref = correspondencias.iloc[0]
        else:
            nivel = "Desvío o correspondencia parcial"
            ref = pd.Series()
        resultados.append({
            "Actividad": act["Descripción de la Actividad"],
            "Unidad": act.get("Unidad", "No especificada"),
            "Nivel de Correspondencia": nivel,
            "Objetivo General": ref.get("Objetivo General", ""),
            "Objetivo Específico": ref.get("Objetivo Específico", ""),
            "Acción PEI": ref.get("Acción", "")
        })

    st.subheader("Resultados de Evaluación")
    resultados_df = pd.DataFrame(resultados)
    st.dataframe(resultados_df)

    output_excel = "evaluacion_consistencia.xlsx"
    resultados_df.to_excel(output_excel, index=False)
    with open(output_excel, "rb") as f:
        st.download_button("Descargar resultados en Excel", f, file_name=output_excel)
else:
    st.info("Por favor, subí un archivo de actividades para comenzar. El archivo debe tener una columna llamada 'Descripción de la Actividad'.")
