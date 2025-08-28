
import io
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st

from utils import (
    normalize_colnames,
    parse_pei_pdf,
    build_plan_index,
    compute_consistency_with_plan,
    compute_pairwise_consistency,
    build_excel_report,
    build_word_report,
    clean_rows,
)

st.set_page_config(page_title="Calculadora PEI – Correlación", layout="wide")
st.title("📊 Calculadora de Consistencia / Correlación – UCCuyo 2023–2027")

mode = st.radio("Modo de correlación", ["Columna 2 ↔ Columna 1 (CSV)", "Contra PDF del PEI"], index=0, help="Elegí si querés correlacionar directamente actividad↔objetivo por archivo (columna 2 vs 1), o usando el PEI en PDF.")

# ----------------------------------
# Carga de archivos de actividades
# ----------------------------------
if "file_bufs" not in st.session_state:
    st.session_state.file_bufs = []
    st.session_state.file_names = []

uploaded = st.file_uploader("📦 Subí 1 o más archivos de actividades (CSV, XLSX, XLS)", type=["csv","xlsx","xls"], accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        if f.name not in st.session_state.file_names:
            st.session_state.file_bufs.append(f.getvalue())
            st.session_state.file_names.append(f.name)
    st.success(f"Se agregaron {len(uploaded)} archivo(s). Total acumulado: {len(st.session_state.file_names)}/6.")

if st.session_state.file_names:
    st.subheader("Archivos acumulados")
    st.write(", ".join(st.session_state.file_names))
    if st.button("🗑️ Limpiar lista"):
        st.session_state.file_bufs = []
        st.session_state.file_names = []
        st.experimental_rerun()

total = len(st.session_state.file_names)
if total < 6:
    st.info("Subí los archivos restantes hasta alcanzar **6** para habilitar el análisis.")
    st.stop()
if total > 6:
    st.error("Hay más de 6 archivos. Limpiá la lista y subí exactamente 6.")
    st.stop()

# Leer archivos
dfs, names = [], []
for name, data in zip(st.session_state.file_names, st.session_state.file_bufs):
    bio = io.BytesIO(data); bio.name = name
    suffix = Path(name).suffix.lower()
    try:
        if suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(bio, engine="openpyxl")
        else:
            try:
                df = pd.read_csv(bio, encoding="utf-8", sep=None, engine="python")
            except Exception:
                bio.seek(0); df = pd.read_csv(bio, encoding="latin-1", sep=None, engine="python")
    except Exception as e:
        st.error(f"❌ No pude leer '{name}'. Detalle: {e}"); st.stop()
    df = normalize_colnames(df)
    df = clean_rows(df)
    dfs.append(df)
    names.append(Path(name).stem)

st.success("✅ Se recibieron 6 archivos correctamente.")
total_actividades = sum(len(df) for df in dfs)
st.info(f"**Total de actividades cargadas (luego de limpiar filas vacías): {total_actividades}**")

# Umbrales
st.subheader("⚙️ Umbrales de clasificación")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena'", min_value=70, max_value=100, value=88, step=1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", min_value=50, max_value=90, value=68, step=1)
thresholds = {"plena": float(t_plena), "parcial": float(t_parcial)}

# ----------------------------------
# MODO A: PDF
# ----------------------------------
if mode == "Contra PDF del PEI":
    pei_pdf = st.file_uploader("📄 Subí el PDF del PEI", type=["pdf"], accept_multiple_files=False, key="pei_pdf")
    if not pei_pdf:
        st.warning("Subí el PDF del PEI para continuar.")
        st.stop()
    with st.spinner("Leyendo y analizando el PEI…"):
        try:
            pei_struct = parse_pei_pdf(pei_pdf)
            plan_index = build_plan_index(pei_struct)
        except Exception as e:
            st.error(f"No pude leer el PDF del PEI. Detalle: {e}")
            st.stop()
    st.success(f"PEI cargado. Objetivos detectados: {len(pei_struct)} | Entradas índice: {len(plan_index)}")
    with st.spinner("Calculando correlación actividad ↔ PEI…"):
        summary_df, detail_tables, matrix_df = compute_consistency_with_plan(dfs, names, plan_index, thresholds)

# ----------------------------------
# MODO B: CSV col2 ↔ col1
# ----------------------------------
else:
    st.subheader("🧩 Selección de columnas")
    cols_map = []
    for i, (name, df) in enumerate(zip(names, dfs), start=1):
        with st.expander(f"{i}. {name}", expanded=(i==1)):
            st.dataframe(df.head(8), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                obj_col = st.selectbox("Columna de **Objetivo** (columna 1)", options=list(df.columns), index=0, key=f"obj_{i}")
            with c2:
                act_col = st.selectbox("Columna de **Actividad** (columna 2)", options=list(df.columns), index=(1 if len(df.columns)>1 else 0), key=f"act_{i}")
            cols_map.append((obj_col, act_col))
    with st.spinner("Calculando correlación columna 2 ↔ columna 1…"):
        summary_df, detail_tables = compute_pairwise_consistency(dfs, names, cols_map, thresholds)
        matrix_df = None  # no aplica

# ----------------------------------
# Resultados + Descargas
# ----------------------------------
st.subheader("📊 Resumen")
st.dataframe(summary_df, use_container_width=True)

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
excel_bytes = build_excel_report(summary_df, detail_tables, labels_used=None, matrix_df=matrix_df)

word_bytes = None
try:
    totals = summary_df[["Total actividades","Consistencia plena","Consistencia parcial","Consistencia nula"]].sum()
    word_bytes = build_word_report(summary_df, totals, names)
except Exception:
    word_bytes = None

c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Descargar Excel (Resumen + Porcentajes + Detalle)",
                       data=excel_bytes,
                       file_name=f"reporte_correlacion_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c2:
    if word_bytes:
        st.download_button("⬇️ Descargar Word (informe narrado)",
                           data=word_bytes,
                           file_name=f"informe_correlacion_{ts}.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.info("No se generó el Word (posible falta de 'python-docx'). Descarga disponible: Excel.")
