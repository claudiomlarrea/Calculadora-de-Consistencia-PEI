import io
import datetime as dt
import pandas as pd
import streamlit as st
from pathlib import Path
from utils import (
    normalize_colnames,
    guess_consistency_column,
    compute_consistency_summary,
    build_excel_report,
    build_word_report,
)

st.set_page_config(page_title="Calculadora de Consistencia PEI – UCCuyo 2023–2027", layout="wide")

st.title("📊 Calculadora de Consistencia PEI – UCCuyo 2023–2027")
st.caption("Secretaría de Investigación – UCCuyo")

with st.expander("¿Cómo funciona?", expanded=False):
    st.markdown("""
    - Podés subir los **6 archivos CSV** de una sola vez o en **varios pasos**.  
    - El sistema detecta la columna de consistencia; si no, podés seleccionarla manualmente.  
    - Genera un **Excel** (resumen + detalle) y un **Word** narrado.
    """)

if "file_bufs" not in st.session_state:
    st.session_state.file_bufs = []
    st.session_state.file_names = []

uploaded = st.file_uploader(
    "Subí 1 o más archivos CSV (hasta completar 6)",
    type=["csv"],
    accept_multiple_files=True,
    help="Podés subirlos en tandas; cuando haya 6, se habilita el análisis."
)

if uploaded:
    for f in uploaded:
        # evitar duplicados por nombre
        if f.name not in st.session_state.file_names:
            st.session_state.file_bufs.append(f.getvalue())
            st.session_state.file_names.append(f.name)
    st.success(f"Se agregaron {len(uploaded)} archivo(s). Total acumulado: {len(st.session_state.file_names)}/6.")

# Mostrar lista acumulada y permitir limpiar
if st.session_state.file_names:
    st.subheader("Archivos acumulados")
    for i, n in enumerate(st.session_state.file_names, start=1):
        st.write(f"{i}. {n}")
    if st.button("🗑️ Limpiar lista"):
        st.session_state.file_bufs = []
        st.session_state.file_names = []
        st.experimental_rerun()

total = len(st.session_state.file_names)

if total < 6:
    st.info("Subí los archivos restantes hasta alcanzar **6** para habilitar el análisis.")
    st.stop()

# --- Procesamiento cuando hay 6 exactos ---
if total > 6:
    st.error("Hay más de 6 archivos. Usá el botón 'Limpiar lista' y volvé a subir exactamente 6.")
    st.stop()

# Reconstruir archivos como UploadedFile-like
files = []
for name, data in zip(st.session_state.file_names, st.session_state.file_bufs):
    files.append(io.BytesIO(data)); files[-1].name = name  # tipo simple con atributo name

dfs = []
names = []
for f in files:
    try:
        df = pd.read_csv(f, encoding="utf-8", sep=None, engine="python")
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, encoding="latin-1", sep=None, engine="python")
    df = normalize_colnames(df)
    dfs.append(df)
    names.append(Path(f.name).stem)

st.success("✅ Se recibieron 6 archivos correctamente.")

st.subheader("🧩 Configuración de consistencia")
default_labels = dict(plena=["plena", "completa", "total"],
                      parcial=["parcial", "media"],
                      nula=["nula", "sin", "no corresponde", "desvío", "desvio", "ninguna"])

col_configs = []
for i, (name, df) in enumerate(zip(names, dfs), start=1):
    with st.expander(f"Archivo {i}: {name}", expanded=(i == 1)):
        col_guess = guess_consistency_column(df)
        selected_col = st.selectbox(
            "Columna de consistencia",
            options=["<ninguna>"] + list(df.columns),
            index=(0 if col_guess is None else list(df.columns).index(col_guess) + 1),
            key=f"col_{i}"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            plena_input = st.text_input("Etiquetas para PLENA (coma-separadas)", ", ".join(default_labels["plena"]), key=f"pl_{i}")
        with c2:
            parcial_input = st.text_input("Etiquetas para PARCIAL (coma-separadas)", ", ".join(default_labels["parcial"]), key=f"pa_{i}")
        with c3:
            nula_input = st.text_input("Etiquetas para NULA (coma-separadas)", ", ".join(default_labels["nula"]), key=f"nu_{i}")

        labels = {
            "plena": [s.strip().lower() for s in plena_input.split(",") if s.strip()],
            "parcial": [s.strip().lower() for s in parcial_input.split(",") if s.strip()],
            "nula": [s.strip().lower() for s in nula_input.split(",") if s.strip()],
        }
        col_configs.append((selected_col if selected_col != "<ninguna>" else None, labels))

# Calcular resumen
summary_rows = []
detail_tables = []
for name, df, (colname, labels) in zip(names, dfs, col_configs):
    result = compute_consistency_summary(df, colname, labels)
    result["objetivo"] = name
    summary_rows.append({
        "Objetivo (archivo)": name,
        "Total actividades": result["total"],
        "Consistencia plena": result["plena"],
        "Consistencia parcial": result["parcial"],
        "Consistencia nula": result["nula"],
        "Sin clasificar": result["sin_clasificar"],
    })
    detail_tables.append((name, result["table"]))

summary_df = pd.DataFrame(summary_rows)
totals = summary_df[["Total actividades", "Consistencia plena", "Consistencia parcial", "Consistencia nula", "Sin clasificar"]].sum()

st.subheader("📊 Resumen")
st.dataframe(summary_df, use_container_width=True)
st.write("**Totales:**", {k:int(v) for k,v in totals.items()})

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
excel_bytes = build_excel_report(summary_df, detail_tables)
word_bytes = build_word_report(summary_df, totals, names)

c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Descargar Excel (resumen + detalle)",
                       data=excel_bytes,
                       file_name=f"reporte_consolidado_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c2:
    st.download_button("⬇️ Descargar Word (informe narrado)",
                       data=word_bytes,
                       file_name=f"informe_consolidado_{ts}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

st.subheader("📄 Detalle por archivo")
for name, table in detail_tables:
    with st.expander(name, expanded=False):
        st.dataframe(table, use_container_width=True)
