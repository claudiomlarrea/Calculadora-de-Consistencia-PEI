import io
import datetime as dt
import pandas as pd
import streamlit as st
from utils import (
    normalize_colnames,
    guess_consistency_column,
    compute_consistency_summary,
    build_excel_report,
    build_word_report,
)

st.set_page_config(page_title="Calculadora de Consistencia PEI – UCCuyo 2023–2027", layout="wide")

st.title("🧮 Calculadora de Consistencia PEI – UCCuyo 2023–2027")
st.caption("Secretaría de Investigación – UCCuyo")

with st.expander("¿Cómo funciona?", expanded=False):
    st.markdown("""
    1) Subí los **6 archivos CSV** (uno por objetivo específico del PEI).  
    2) El sistema intentará **detectar automáticamente** la columna que indica el tipo de consistencia.  
    3) Si es necesario, **seleccioná manualmente** la columna y las etiquetas para **Plena / Parcial / Nula**.  
    4) Descargá los informes **Excel** y **Word** generados.
    """)

uploaded = st.file_uploader(
    "Subí exactamente **6 archivos CSV** (uno por cada objetivo específico)",
    type=["csv"],
    accept_multiple_files=True,
    help="Cada archivo debe corresponder a un objetivo específico del PEI",
)

if not uploaded:
    st.info("Esperando archivos…")
    st.stop()

if len(uploaded) != 6:
    st.error(f"Se recibieron {len(uploaded)} archivo(s). Deben ser exactamente 6.")
    st.stop()

# Leer y normalizar
dfs = []
names = []

for f in uploaded:
    try:
        df = pd.read_csv(f, encoding="utf-8", sep=None, engine="python")
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, encoding="latin-1", sep=None, engine="python")
    df = normalize_colnames(df)
    dfs.append(df)
    # nombre corto sin extensión para reporte
    names.append(Path(f.name).stem)

st.success("✅ 6 archivos cargados correctamente.")

# Detección de columna de consistencia (por cada archivo se puede elegir)
st.subheader("🧩 Configuración de consistencia")
st.write("Seleccioná la columna que clasifica cada actividad como **Plena / Parcial / Nula**. Podés ajustar etiquetas si tus archivos usan otros términos.")

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

        # Etiquetas para mapear a plena/parcial/nula
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
st.write("**Totales:**", dict(totals))

# Construir archivos para descarga
ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
excel_bytes = build_excel_report(summary_df, detail_tables)
word_bytes = build_word_report(summary_df, totals, names)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Descargar Excel (resumen + detalle)",
        data=excel_bytes,
        file_name=f"reporte_consolidado_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with c2:
    st.download_button(
        "⬇️ Descargar Word (informe narrado)",
        data=word_bytes,
        file_name=f"informe_consolidado_{ts}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

st.success("Listo. También podés revisar cada tabla de detalle en la sección siguiente.")

st.subheader("📄 Detalle por archivo")
for name, table in detail_tables:
    with st.expander(name, expanded=False):
        st.dataframe(table, use_container_width=True)

st.caption("No se necesita un archivo de referencia adicional. Subí únicamente los 6 CSV.")
