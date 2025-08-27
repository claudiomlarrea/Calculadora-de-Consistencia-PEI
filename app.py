
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO
from docx import Document

st.set_page_config(page_title="Calculadora de Consistencia PEI – UCCuyo 2023–2027", layout="wide")
st.title("📊 Calculadora de Consistencia PEI – UCCuyo 2023–2027")
st.markdown("Subí los **6 archivos de objetivos del PEI** (uno por objetivo). La calculadora unificará los archivos, analizará las actividades institucionales y te devolverá un Excel con los porcentajes de coherencia y un informe narrado en Word.")

uploaded_files = st.file_uploader("📂 Cargar los 6 archivos de objetivos del PEI (formato CSV)", type="csv", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 6:
    st.success("✅ Archivos cargados correctamente. Procesando...")

    # Unificar todos los archivos en un solo DataFrame
    df_total = pd.concat([pd.read_csv(file) for file in uploaded_files], ignore_index=True)
    df_total.fillna("", inplace=True)

    # Analizar consistencia (ejemplo básico: buscar si palabras clave del objetivo están en la actividad)
    resultados = []
    for _, row in df_total.iterrows():
        objetivo = str(row.get("Objetivo Específico", ""))
        actividad = str(row.get("Actividad", ""))
        if any(palabra.lower() in actividad.lower() for palabra in objetivo.split()):
            consistencia = "Plena"
        elif any(palabra.lower()[:5] in actividad.lower() for palabra in objetivo.split()):
            consistencia = "Parcial"
        else:
            consistencia = "Nula"
        resultados.append((objetivo, actividad, consistencia))

    df_resultados = pd.DataFrame(resultados, columns=["Objetivo", "Actividad", "Consistencia"])
    resumen = df_resultados["Consistencia"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%"

    # Guardar Excel con resultados
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df_resultados.to_excel(writer, index=False, sheet_name="Análisis de consistencia")
        pd.DataFrame(resumen).to_excel(writer, sheet_name="Resumen porcentual")
    excel_output.seek(0)

    # Crear documento Word
    doc = Document()
    doc.add_heading("Informe de Análisis de Consistencia del PEI", level=1)
    doc.add_paragraph("A continuación se detallan los resultados del análisis de coherencia entre las actividades institucionales y los objetivos específicos del PEI 2023–2027 de la UCCuyo.")

    for tipo in ["Plena", "Parcial", "Nula"]:
        count = (df_resultados["Consistencia"] == tipo).sum()
        doc.add_heading(f"Consistencia {tipo}", level=2)
        doc.add_paragraph(f"Cantidad de actividades con consistencia {tipo.lower()}: {count}")

    doc_output = BytesIO()
    doc.save(doc_output)
    doc_output.seek(0)

    # Ofrecer archivos para descarga
    st.download_button("📥 Descargar Excel con análisis", data=excel_output, file_name="reporte_analisis_PEI.xlsx")
    st.download_button("📥 Descargar informe Word narrado", data=doc_output, file_name="informe_analisis_PEI.docx")
else:
    st.info("📌 Subí exactamente 6 archivos CSV correspondientes a los objetivos del PEI.")
