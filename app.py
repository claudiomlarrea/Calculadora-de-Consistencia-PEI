
import io
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st

from utils import (
    normalize_colnames, clean_rows, detect_columns, count_valid_pairs,
    compute_pairwise_consistency_single,
    parse_pei_pdf, build_plan_index, compute_consistency_pei_single,
    excel_from_blocks
)

st.set_page_config(page_title="Formulario Único – Consistencias", layout="wide")
st.title("📊 Consistencias a partir del Formulario Único (280 actividades)")

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
st.subheader("Seleccioná columnas")
c1, c2 = st.columns(2)
with c1:
    col_obj = st.selectbox("Columna de **Objetivo específico**", options=list(df.columns), index=(list(df.columns).index(obj_default) if obj_default in df.columns else 0))
with c2:
    col_act = st.selectbox("Columna de **Actividad**", options=list(df.columns), index=(list(df.columns).index(act_default) if act_default in df.columns else (1 if len(df.columns)>1 else 0)))

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

# ------------ A) Consistencia independiente -------------
sum_indep, det_indep = compute_pairwise_consistency_single(df, uploaded.name, col_obj, col_act, thr)
resumen_indep = pd.DataFrame([sum_indep])
porc_indep = resumen_indep.copy()
for col in ["Consistencia plena","Consistencia parcial","Consistencia nula"]:
    porc_indep[f"% {col}"] = (porc_indep[col] / porc_indep["Total actividades"]).round(4)
porc_indep = porc_indep[["Fuente", "% Consistencia plena","% Consistencia parcial","% Consistencia nula"]]

# ------------ B) Consistencia contra PEI (opcional) -------------
st.subheader("Contra el PEI (opcional)")
pei_pdf = st.file_uploader("Subí el PDF del PEI (opcional)", type=["pdf"], key="pei")
if pei_pdf:
    with st.spinner("Analizando PEI…"):
        pei = parse_pei_pdf(pei_pdf)
        index = build_plan_index(pei)
    sum_pei, det_pei = compute_consistency_pei_single(df, uploaded.name, col_act, index, thr)
    resumen_pei = pd.DataFrame([sum_pei])
    porc_pei = resumen_pei.copy()
    for col in ["Consistencia plena","Consistencia parcial","Consistencia nula"]:
        porc_pei[f"% {col}"] = (porc_pei[col] / porc_pei["Total actividades"]).round(4)
    porc_pei = porc_pei[["Fuente", "% Consistencia plena","% Consistencia parcial","% Consistencia nula"]]
else:
    resumen_pei = None; porc_pei = None; det_pei = None

# ------------ Descargas -------------
ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
blocks = [
    ("Resumen_independiente", pd.DataFrame([sum_indep])),
    ("Porcentajes_independiente", porc_indep),
    ("Detalle_independiente", det_indep),
]
if resumen_pei is not None:
    blocks.extend([
        ("Resumen_PEI", resumen_pei),
        ("Porcentajes_PEI", porc_pei),
        ("Detalle_PEI", det_pei),
    ])

excel_bytes = excel_from_blocks(blocks)
st.download_button("⬇️ Descargar Excel (independiente + PEI si se cargó)", data=excel_bytes, file_name=f"consistencias_formulario_unico_{ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
