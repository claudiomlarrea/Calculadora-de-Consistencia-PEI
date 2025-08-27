import streamlit as st
import pandas as pd
import io
import base64
from docx import Document
from openpyxl import Workbook

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="wide")
st.title("🧮 Calculadora de Consistencia PEI – UCCuyo 2023–2027")

st.markdown("Subí el archivo `pei_referencia.csv` consolidado con todos los objetivos del PEI y el archivo de actividades del Formulario Único.")

uploaded_pei = st.file_uploader("📑 Cargar archivo de referencia PEI (pei_referencia.csv)", type="csv")
uploaded_actividades = st.file_uploader("📋 Cargar archivo de actividades institucionales", type="csv")

if uploaded_pei and uploaded_actividades:
    df_pei = pd.read_csv(uploaded_pei)
    df_acts = pd.read_csv(uploaded_actividades)

    df_pei['Objetivos_Unificados'] = df_pei[['Objetivo General', 'Objetivo Específico', 'Acciones Estratégicas']].fillna('').agg(' '.join, axis=1)
    df_acts['Coincidencias'] = df_acts['Descripción de la Actividad'].apply(
        lambda x: sum(1 for palabra in df_pei['Objetivos_Unificados'] if str(x).lower() in str(palabra).lower())
    )
    df_acts['Nivel de Coherencia'] = df_acts['Coincidencias'].apply(
        lambda x: "Alta" if x >= 3 else ("Media" if x == 2 else ("Baja" if x == 1 else "Sin correspondencia"))
    )

    st.success("✅ Análisis realizado correctamente")
    st.dataframe(df_acts[['Unidad Académica', 'Descripción de la Actividad', 'Nivel de Coherencia']])

    resumen = df_acts['Nivel de Coherencia'].value_counts().rename_axis('Nivel').reset_index(name='Cantidad')
    st.subheader("📊 Resumen del Análisis")
    st.table(resumen)

    st.markdown("---")

    # Generar Excel
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_acts.to_excel(writer, sheet_name='Análisis', index=False)
        resumen.to_excel(writer, sheet_name='Resumen', index=False)
    output_excel.seek(0)
    b64_excel = base64.b64encode(output_excel.read()).decode()
    href_excel = f'<a href="data:application/octet-stream;base64,{b64_excel}" download="reporte_analisis_PEI.xlsx">📥 Descargar Informe en Excel</a>'
    st.markdown(href_excel, unsafe_allow_html=True)

    # Generar Word
    doc = Document()
    doc.add_heading('Informe de Consistencia PEI – UCCuyo 2023–2027', level=1)
    doc.add_paragraph("Resumen de niveles de coherencia detectados en las actividades institucionales:")
    for _, row in resumen.iterrows():
        doc.add_paragraph(f"- {row['Nivel']}: {row['Cantidad']} actividades")
    doc.add_paragraph("Se recomienda revisar las actividades marcadas como 'Sin correspondencia' o con 'Baja' coherencia para mejorar su alineación con el PEI.")
    output_word = io.BytesIO()
    doc.save(output_word)
    output_word.seek(0)
    b64_word = base64.b64encode(output_word.read()).decode()
    href_word = f'<a href="data:application/octet-stream;base64,{b64_word}" download="informe_analisis_PEI.docx">📥 Descargar Informe en Word</a>'
    st.markdown(href_word, unsafe_allow_html=True)
