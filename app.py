
import streamlit as st
from pathlib import Path
import pandas as pd
import io
from docx import Document

st.set_page_config(page_title="Calculadora de Consistencia PEI", layout="centered")
st.title("📊 Calculadora de Consistencia PEI – UCCuyo 2023–2027")
st.markdown("Subí los 6 archivos CSV correspondientes a cada objetivo del PEI. La aplicación procesará automáticamente los datos y generará un Excel con análisis y un informe Word narrado.")

uploaded_files = st.file_uploader("📎 Cargar los 6 archivos de objetivos del PEI (formato CSV)", type="csv", accept_multiple_files=True)

def analizar_consistencia(dataframes):
    resumen = []
    for df in dataframes:
        nombre = df.name
        df_data = pd.read_csv(df)
        total = len(df_data)
        consistencia_plena = df_data[df_data['Consistencia'].str.lower() == 'plena']
        consistencia_parcial = df_data[df_data['Consistencia'].str.lower() == 'parcial']
        consistencia_nula = df_data[df_data['Consistencia'].str.lower() == 'nula']
        resumen.append({
            "Archivo": nombre,
            "Total Actividades": total,
            "Consistencia Plena": len(consistencia_plena),
            "Consistencia Parcial": len(consistencia_parcial),
            "Consistencia Nula": len(consistencia_nula)
        })
    return pd.DataFrame(resumen)

def generar_reporte_narrado(df_resumen):
    doc = Document()
    doc.add_heading("Informe de Análisis de Consistencia del PEI", level=1)
    doc.add_paragraph("A continuación se detallan los resultados del análisis de coherencia entre las actividades institucionales y los objetivos específicos del PEI 2023–2027 de la UCCuyo.")
    for index, row in df_resumen.iterrows():
        doc.add_heading(row["Archivo"], level=2)
        doc.add_paragraph(f"Total de actividades analizadas: {row['Total Actividades']}")
        doc.add_paragraph(f"Actividades con consistencia plena: {row['Consistencia Plena']}")
        doc.add_paragraph(f"Actividades con consistencia parcial: {row['Consistencia Parcial']}")
        doc.add_paragraph(f"Actividades con consistencia nula: {row['Consistencia Nula']}")
    output = io.BytesIO()
    doc.save(output)
    return output

if uploaded_files:
    st.success("✅ Archivos cargados correctamente. Procesando...")
    resumen_df = analizar_consistencia(uploaded_files)
    excel_output = io.BytesIO()
    resumen_df.to_excel(excel_output, index=False)
    word_output = generar_reporte_narrado(resumen_df)

    st.download_button("📥 Descargar Excel con análisis", data=excel_output.getvalue(), file_name="reporte_analisis_PEI.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📥 Descargar informe Word narrado", data=word_output.getvalue(), file_name="informe_analisis_PEI.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
