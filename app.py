import io
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
from docx import Document
from docx.shared import Inches

from utils import (
    normalize_colnames, clean_rows, detect_columns, count_valid_pairs,
    compute_pairwise_consistency_single,
    parse_pei_pdf, build_plan_index, compute_consistency_pei_single,
    excel_from_blocks
)
from advanced_analysis import generate_advanced_report, format_report_for_display

st.set_page_config(page_title="Análisis Avanzado de Consistencia PEI", layout="wide")
st.title("📊 Análisis Avanzado de Consistencia - Formulario Único PEI")

# ------------ Carga del archivo único -------------
uploaded = st.file_uploader("Subí el **Formulario Único** (XLSX o CSV)", type=["xlsx","csv"])
if not uploaded:
    st.stop()

# leer archivo
bio = io.BytesIO(uploaded.getvalue()); bio.name = uploaded.name
if uploaded.name.lower().endswith(".xlsx"):
    df = pd.read_excel(bio, engine="openpyxl")
else:
    try:
        df = pd.read_csv(bio, encoding="utf-8", sep=None, engine="python")
    except Exception:
        bio.seek(0); df = pd.read_csv(bio, encoding="latin-1", sep=None, engine="python")

df = normalize_colnames(df)
df = clean_rows(df)

# detectar columnas y permitir cambio
obj_default, act_default = detect_columns(df)
st.subheader("Selecciona columnas")
c1, c2 = st.columns(2)
with c1:
    col_obj = st.selectbox("Columna de **Objetivo específico**", options=list(df.columns), 
                          index=(list(df.columns).index(obj_default) if obj_default in df.columns else 0))
with c2:
    col_act = st.selectbox("Columna de **Actividad**", options=list(df.columns), 
                          index=(list(df.columns).index(act_default) if act_default in df.columns else (1 if len(df.columns)>1 else 0)))

total_valid = count_valid_pairs(df, col_obj, col_act)
st.info(f"**Total de actividades (objetivo & actividad no vacíos): {total_valid}**")

st.subheader("Previsualización")
st.dataframe(df[[col_obj, col_act]].head(12), use_container_width=True)

# ------------ Umbrales -------------
st.subheader("Umbrales")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena'", 70, 100, 88, 1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", 50, 90, 68, 1)
thr = {"plena": float(t_plena), "parcial": float(t_parcial)}

# ------------ Análisis de consistencia -------------
if st.button("🔍 Realizar Análisis Avanzado", type="primary"):
    with st.spinner("Calculando consistencias..."):
        sum_indep, det_indep = compute_pairwise_consistency_single(df, uploaded.name, col_obj, col_act, thr)
        
        # Generar análisis avanzado
        advanced_report = generate_advanced_report(det_indep, col_obj, col_act)
        
        # Mostrar resumen ejecutivo
        st.markdown("---")
        st.markdown(format_report_for_display(advanced_report))
        
        # Mostrar tablas detalladas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Objetivos Críticos (Top 10)")
            if not advanced_report['rendimiento_objetivos'].empty:
                st.dataframe(advanced_report['rendimiento_objetivos'], use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar objetivos críticos")
        
        with col2:
            st.subheader("📈 Mayor Dispersión Interna")
            if not advanced_report['dispersion_objetivos'].empty:
                st.dataframe(advanced_report['dispersion_objetivos'], use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar dispersión")
        
        st.subheader("🔄 Actividades con Potencial de Mejora/Reubicación")
        if not advanced_report['potencial_mejora'].empty:
            st.dataframe(advanced_report['potencial_mejora'], use_container_width=True)
            st.caption("💡 **Nota**: La reubicación debe considerarse luego de reelaborar la redacción de la actividad.")
        else:
            st.info("No se encontraron actividades con alto potencial de mejora")
        
        st.subheader("🔍 Actividades Duplicadas o Similares")
        if not advanced_report['duplicados'].empty:
            st.dataframe(advanced_report['duplicados'], use_container_width=True)
            st.caption("💡 **Sugerencia**: Consolidar duplicadas como líneas de trabajo con sub-tareas medibles")
        else:
            st.success("No se detectaron actividades duplicadas")
        
        # Recomendaciones
        st.subheader("📋 Plan de Mejora por Etapas")
        recs = advanced_report['recomendaciones']
        
        with st.expander("🟡 Corto Plazo (0-30 días)", expanded=True):
            for rec in recs['corto_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("🟠 Mediano Plazo (1-3 meses)"):
            for rec in recs['mediano_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("🔴 Largo Plazo (3+ meses)"):
            for rec in recs['largo_plazo']:
                st.write(f"• {rec}")
        
        # ------------ Guía de Reescritura -------------
        st.subheader("✍️ Guía Práctica de Reescritura")
        st.markdown("""
        **Plantilla sugerida para objetivos y actividades:**
        
        `Verbo operativo + Objeto + Ámbito/Población + Entregable + Resultado esperado`
        
        **Ejemplos:**
        - ✅ **Buena**: "Implementar tablero de seguimiento en Looker Studio para objetivos que contemple actualización mensual → Indicadores disponibles y monitoreados"
        - ✅ **Buena**: "Diseñar e institucionalizar protocolo de autoevaluación anual → Informes de autoevaluación y plan de mejora"
        - ❌ **Mala**: "Realizar actividades de capacitación"
        """)

# ------------ Contra el PEI (opcional) -------------
st.subheader("🎯 Análisis contra el PEI (opcional)")
pei_pdf = st.file_uploader("Subí el PDF del PEI (opcional)", type=["pdf"], key="pei")
if pei_pdf:
    with st.spinner("Analizando PEI..."):
        pei = parse_pei_pdf(pei_pdf)
        index = build_plan_index(pei)
    sum_pei, det_pei = compute_consistency_pei_single(df, uploaded.name, col_act, index, thr)
    
    st.success(f"Análisis contra PEI completado: {sum_pei['Total actividades']} actividades evaluadas")
    
    # Mostrar resultados PEI
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consistencia Plena", sum_pei['Consistencia plena'], 
                 f"{sum_pei['Consistencia plena']/sum_pei['Total actividades']*100:.1f}%")
    with col2:
        st.metric("Consistencia Parcial", sum_pei['Consistencia parcial'], 
                 f"{sum_pei['Consistencia parcial']/sum_pei['Total actividades']*100:.1f}%")
    with col3:
        st.metric("Consistencia Nula", sum_pei['Consistencia nula'], 
                 f"{sum_pei['Consistencia nula']/sum_pei['Total actividades']*100:.1f}%")

# ------------ Descargas -------------
if 'det_indep' in locals():
    st.subheader("💾 Descargas")
    
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Preparar bloques para Excel
    blocks = [
        ("Resumen_Estadisticas", pd.DataFrame([advanced_report['estadisticas_generales']])),
        ("Distribucion_Niveles", pd.DataFrame([advanced_report['distribucion_niveles']])),
        ("Objetivos_Criticos", advanced_report['rendimiento_objetivos']),
        ("Dispersion_Objetivos", advanced_report['dispersion_objetivos']),
        ("Potencial_Mejora", advanced_report['potencial_mejora']),
        ("Actividades_Duplicadas", advanced_report['duplicados']),
        ("Detalle_Completo", det_indep),
    ]
    
    if 'det_pei' in locals():
        blocks.extend([
            ("Resumen_PEI", pd.DataFrame([sum_pei])),
            ("Detalle_PEI", det_pei),
        ])
    
    # Generar Excel
    excel_bytes = excel_from_blocks(blocks)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Descargar Excel Completo", 
            data=excel_bytes, 
            file_name=f"analisis_consistencia_avanzado_{ts}.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Generar informe Word
        word_bytes = generate_word_report(advanced_report, uploaded.name, ts)
        st.download_button(
            "⬇️ Descargar Informe Word", 
            data=word_bytes, 
            file_name=f"informe_consistencia_pei_{ts}.docx", 
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

def generate_word_report(report: dict, filename: str, timestamp: str) -> bytes:
    """Genera un informe en formato Word"""
    doc = Document()
    
    # Título
    title = doc.add_heading('Informe de Consistencia PEI', 0)
    
    # Información del archivo
    doc.add_paragraph(f"Archivo procesado: {filename}")
    doc.add_paragraph(f"Fecha de análisis: {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    doc.add_paragraph("")
    
    # Estadísticas generales
    stats = report['estadisticas_generales']
    doc.add_heading('Estadísticas Generales', level=1)
    doc.add_paragraph(f"Cantidad de actividades evaluadas: {stats['total_actividades']}")
    doc.add_paragraph(f"Porcentaje promedio de consistencia general: {stats['promedio_consistencia']}%")
    doc.add_paragraph(f"Mediana: {stats['mediana']}% | P25: {stats['p25']}% | P75: {stats['p75']}% | Mín/Máx: {stats['minimo']}% / {stats['maximo']}%")
    doc.add_paragraph("")
    
    # Distribución por niveles
    dist = report['distribucion_niveles']
    doc.add_heading('Distribución por niveles', level=1)
    
    table = doc.add_table(rows=4, cols=2)
    table.style = 'Table Grid'
    
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Nivel'
    hdr_cells[1].text = 'N actividades'
    
    table.rows[1].cells[0].text = 'Alta (≥75%)'
    table.rows[1].cells[1].text = str(dist['alta_75_plus'])
    
    table.rows[2].cells[0].text = 'Media (50-74%)'
    table.rows[2].cells[1].text = str(dist['media_50_74'])
    
    table.rows[3].cells[0].text = 'Baja (<50%)'
    table.rows[3].cells[1].text = str(dist['baja_menor_50'])
    
    doc.add_paragraph("")
    doc.add_paragraph("Interpretación: una mayor proporción en niveles Medio/Alto sugiere redacciones alineadas con los verbos, ámbitos y productos de los objetivos. Una concentración en Bajo indica redacciones genéricas u objetivos poco acotados.")
    doc.add_paragraph("")
    
    # Recomendaciones
    doc.add_heading('Plan de mejora por etapas', level=1)
    
    doc.add_heading('Corto plazo (0-30 días)', level=2)
    for rec in report['recomendaciones']['corto_plazo']:
        p = doc.add_paragraph()
        p.style = 'List Bullet'
        p.text = rec
    
    doc.add_heading('Mediano plazo (1-3 meses)', level=2)
    for rec in report['recomendaciones']['mediano_plazo']:
        p = doc.add_paragraph()
        p.style = 'List Bullet'
        p.text = rec
    
    doc.add_heading('Largo plazo (3+ meses)', level=2)
    for rec in report['recomendaciones']['largo_plazo']:
        p = doc.add_paragraph()
        p.style = 'List Bullet'
        p.text = rec
    
    # Guardar en memoria
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
