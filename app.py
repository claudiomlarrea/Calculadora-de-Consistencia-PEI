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
    build_excel_report,
    build_word_report,
    clean_rows,
)

st.set_page_config(page_title="Calculadora de Consistencia PEI – UCCuyo 2023–2027 (correlación con PEI)",
                   layout="wide")

st.title("📊 Calculadora de Consistencia PEI – UCCuyo 2023–2027 (correlación con PEI)")
st.caption("Carga 6 archivos de actividades (CSV/XLSX) + el PDF del Plan Estratégico para correlacionar actividad ↔ objetivo/acción/indicador.")

with st.expander("Instrucciones", expanded=False):
    st.markdown("""
    1) Subí **el PDF del Plan Estratégico** (PEI).  
    2) Subí los **6 archivos** de actividades (CSV, XLSX o XLS) —podés hacerlo **en tandas**; la app acumula hasta 6.  
    3) Ajustá los **umbrales** de clasificación si querés ser más estricto.  
    4) Descargá **Excel** (Resumen + Porcentajes + Matriz + Detalle) y **Word** narrado.
    """)

# --- Sección: cargar PEI (PDF) ---
pei_pdf = st.file_uploader("📄 Subí el PDF del Plan Estratégico Institucional (PEI)", type=["pdf"], accept_multiple_files=False)
plan_index = None
if pei_pdf:
    with st.spinner("Leyendo y analizando el PEI…"):
        try:
            pei_struct = parse_pei_pdf(pei_pdf)
            plan_index = build_plan_index(pei_struct)
        except Exception as e:
            st.error(f"No pude leer el PDF del PEI. Detalle: {e}")
            st.stop()
    st.success(f"PEI cargado. Objetivos detectados: {len(pei_struct)} | Entradas en índice: {len(plan_index)}")
    with st.expander("Ver índice (primeras 10 entradas)", expanded=False):
        if plan_index:
            st.dataframe(pd.DataFrame(plan_index).head(10), use_container_width=True)

if not plan_index:
    st.info("Subí primero el **PDF del PEI** para poder correlacionar actividades con Objetivos/Acciones/Indicadores.")
    st.stop()

# --- Sección: carga de archivos de actividades ---
if "file_bufs" not in st.session_state:
    st.session_state.file_bufs = []
    st.session_state.file_names = []

uploaded = st.file_uploader("📦 Subí 1 o más archivos de actividades (CSV, XLSX, XLS)",
                            type=["csv","xlsx","xls"],
                            accept_multiple_files=True)

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

# --- Conteo exacto de actividades ---
total_actividades = sum(len(df) for df in dfs)
st.info(f"**Total de actividades cargadas (luego de limpiar filas vacías): {total_actividades}**")

# --- Umbrales de clasificación (conservadores) ---
st.subheader("⚙️ Umbrales de clasificación (conservadores)")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena' (requiere además coherencia de objetivo si hay pista)", min_value=70, max_value=100, value=88, step=1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", min_value=50, max_value=90, value=68, step=1)

thresholds = {"plena": float(t_plena), "parcial": float(t_parcial)}

# --- Cálculo de correlación exacta contra PEI ---
with st.spinner("Calculando correlación actividad ↔ PEI…"):
    summary_df, detail_tables, matrix_df = compute_consistency_with_plan(dfs, names, plan_index, thresholds)

st.subheader("📊 Resumen")
st.dataframe(summary_df, use_container_width=True)

if not matrix_df.empty:
    st.subheader("🧮 Matriz por OBJETIVO (conteos por categoría)")
    st.dataframe(matrix_df, use_container_width=True)

# Descargas
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
    st.download_button("⬇️ Descargar Excel (Resumen+Porcentajes+Matriz+Detalle)",
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
