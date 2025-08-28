
import io
import datetime as dt
import pandas as pd
import streamlit as st

from utils import (
    normalize_colnames, clean_rows, detect_columns, count_valid_pairs,
    analyze_independent, excel_consolidado, docx_conclusiones
)

st.set_page_config(page_title="Calculadora – Formulario Único", layout="wide")
st.title("📊 Calculadora de Consistencia – Formulario Único")

uploaded = st.file_uploader("Subí el **Formulario Único** (XLSX o CSV)", type=["xlsx","csv"])
if not uploaded:
    st.stop()

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

obj_default, act_default = detect_columns(df)
st.subheader("Seleccioná columnas")
c1, c2 = st.columns(2)
with c1:
    col_obj = st.selectbox("Columna de **Objetivo específico**", options=list(df.columns),
                           index=(list(df.columns).index(obj_default) if obj_default in df.columns else 0))
with c2:
    col_act = st.selectbox("Columna de **Actividad**", options=list(df.columns),
                           index=(list(df.columns).index(act_default) if act_default in df.columns else (1 if len(df.columns)>1 else 0)))

total_valid = count_valid_pairs(df, col_obj, col_act)
st.info(f"Total de actividades válidas (objetivo + actividad): **{total_valid}**")

st.subheader("Vista previa")
st.dataframe(df[[col_obj, col_act]].head(15), use_container_width=True)

st.subheader("Umbrales de clasificación")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena'", 70, 100, 88, 1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", 50, 90, 68, 1)

if st.button("🔎 Realizar Análisis Completo de Consistencia"):
    with st.spinner("Calculando consistencias y generando informes…"):
        resumen, detalle, cons_obj, mejoras, dup = analyze_independent(
            df, uploaded.name, col_obj, col_act, thr_full=float(t_plena), thr_partial=float(t_parcial)
        )

    st.subheader("📊 Resumen")
    st.write(resumen)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_bytes = excel_consolidado(resumen, detalle, cons_obj, mejoras, dup)
    word_bytes  = docx_conclusiones(resumen, detalle, cons_obj, mejoras)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Descargar EXCEL — informe_consistencia_pei_consolidado",
                           data=excel_bytes,
                           file_name=f"informe_consistencia_pei_consolidado_{ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        if word_bytes:
            st.download_button("⬇️ Descargar WORD — Diagnóstico Completo del Formulario",
                               data=word_bytes,
                               file_name=f'Diagnostico_Completo_del_Formulario_{ts}.docx',
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.info("No se pudo generar el Word (falta python-docx). El Excel está disponible.")
else:
    st.warning("Cargá el archivo y tocá el botón para generar los informes.")
