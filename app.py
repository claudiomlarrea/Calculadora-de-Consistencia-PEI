
import io
import re
import unicodedata
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from docx import Document

# ---------------------------
# Utilidades de limpieza
# ---------------------------
SPANISH_STOPWORDS = {
    "a","ante","bajo","cabe","con","contra","de","desde","durante","en","entre","hacia","hasta","mediante",
    "para","por","según","sin","so","sobre","tras","el","la","los","las","un","una","unos","unas","y","o",
    "u","e","ni","que","como","al","del","se","su","sus","es","son","ser","estar","esta","este","estos","estas",
    "hay","más","menos","muy","ya","no","sí","si","pero","porque","cuando","donde","entre","sobre","cada",
    "lo","le","les","debe","deben","deber","deberá","deberán","debería","deberían","puede","pueden","podrá",
    "podrán","podría","podrían","también","además"
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    return [t for t in normalize_text(s).split() if t and t not in SPANISH_STOPWORDS]

def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def combined_score(activity: str, objective: str) -> float:
    t_ratio = fuzz.token_set_ratio(activity, objective) / 100.0
    jac = jaccard(tokens(activity), tokens(objective))
    return float(0.6 * t_ratio + 0.4 * jac)

def band(score: float) -> str:
    pct = score * 100
    if pct >= 75: return "Alta"
    if pct >= 50: return "Media"
    return "Baja"

def best_column(df: pd.DataFrame, candidates):
    best, best_score = None, -1
    for col in df.columns:
        col_norm = normalize_text(col)
        for cand in candidates:
            s = fuzz.partial_ratio(col_norm, normalize_text(cand))
            if s > best_score:
                best, best_score = col, s
    return best

def guess_columns(df: pd.DataFrame):
    col_obj = best_column(df, ["objetivo especifico", "objetivo", "objetivo pei"])
    col_act = best_column(df, ["actividad", "acciones", "descripcion de la actividad", "accion"])
    col_uni = best_column(df, ["unidad academica", "unidad", "facultad", "instituto", "secretaria"])
    col_resp = best_column(df, ["responsable", "responsables", "area responsable"])
    return col_obj, col_act, col_uni, col_resp

def is_blank(x: str) -> bool:
    return normalize_text(x) == ""

def evaluate(df: pd.DataFrame, col_obj: str, col_act: str, col_uni: Optional[str], col_resp: Optional[str]) -> pd.DataFrame:
    work = df.copy()
    work["_objetivo"] = work[col_obj].astype(str)
    work["_actividad"] = work[col_act].astype(str)
    if col_uni: work["_unidad"] = work[col_uni].astype(str)
    if col_resp: work["_responsable"] = work[col_resp].astype(str)

    # Tratar objetivos vacíos: etiquetar y score 0
    vacio_mask = work["_objetivo"].apply(is_blank)
    work.loc[vacio_mask, "_objetivo"] = "Sin objetivo (vacío)"
    # Calcular score normalmente y luego forzar 0 a vacíos
    work["score"] = work.apply(lambda r: combined_score(r["_actividad"], r["_objetivo"]), axis=1)
    work.loc[work["_objetivo"] == "Sin objetivo (vacío)", "score"] = 0.0

    work["consistencia_%"] = (work["score"] * 100).round(1)
    work["nivel"] = work["score"].apply(band)

    res_cols = ["consistencia_%","nivel"]
    if col_uni: res_cols.append("_unidad")
    if col_resp: res_cols.append("_responsable")
    res_cols.extend(["_objetivo","_actividad"])
    out = work[res_cols].rename(columns={
        "_unidad":"Unidad Académica",
        "_responsable":"Responsable",
        "_objetivo":"Objetivo específico",
        "_actividad":"Actividad"
    })
    return out

def aggregations(out: pd.DataFrame):
    by_unidad = None
    if "Unidad Académica" in out.columns:
        by_unidad = out.groupby("Unidad Académica", dropna=False)["consistencia_%"].agg(["count","mean"]).reset_index()
        by_unidad = by_unidad.rename(columns={"count":"N actividades","mean":"Promedio %"})
        by_unidad["Promedio %"] = by_unidad["Promedio %"].round(1)
    by_obj = out.groupby("Objetivo específico", dropna=False)["consistencia_%"].agg(["count","mean"]).reset_index()
    by_obj = by_obj.rename(columns={"count":"N actividades","mean":"Promedio %"})
    by_obj["Promedio %"] = by_obj["Promedio %"].round(1)
    return by_unidad, by_obj

def generar_informe_word(promedio_excl, out_df, resumen_obj, resumen_uni=None, n_excluidos=0):
    doc = Document()
    doc.add_heading("Análisis de consistencia de actividades PEI", 0)
    doc.add_paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y')}")
    doc.add_paragraph(f"Promedio general de consistencia (excluye objetivos vacíos): {promedio_excl:.1f}%")
    if n_excluidos > 0:
        doc.add_paragraph(f"Actividades con objetivo vacío (marcadas como 'Sin objetivo (vacío)'): {n_excluidos}")

    doc.add_heading("Resumen por objetivo específico", level=1)
    t = doc.add_table(rows=1, cols=len(resumen_obj.columns))
    t.style = "Light List Accent 1"
    for j, col in enumerate(resumen_obj.columns):
        t.cell(0,j).text = str(col)
    for _, row in resumen_obj.iterrows():
        cells = t.add_row().cells
        for j, col in enumerate(resumen_obj.columns):
            cells[j].text = str(row[col])

    if resumen_uni is not None:
        doc.add_heading("Resumen por unidad académica", level=1)
        t2 = doc.add_table(rows=1, cols=len(resumen_uni.columns))
        t2.style = "Light List Accent 1"
        for j, col in enumerate(resumen_uni.columns):
            t2.cell(0,j).text = str(col)
        for _, row in resumen_uni.iterrows():
            cells = t2.add_row().cells
            for j, col in enumerate(resumen_uni.columns):
                cells[j].text = str(row[col])

    doc.add_heading("Conclusiones", level=1)
    if promedio_excl >= 75:
        doc.add_paragraph("El nivel de consistencia global es alto, con adecuada alineación entre actividades y objetivos específicos.")
    elif promedio_excl >= 50:
        doc.add_paragraph("El nivel de consistencia global es medio; se recomienda revisar objetivos con promedios bajos y casos 'Sin objetivo (vacío)'.")
    else:
        doc.add_paragraph("El nivel de consistencia global es bajo; se sugiere reestructurar actividades para mejorar su contribución a los objetivos específicos.")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Análisis de consistencia de actividades PEI", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI")
st.caption("Calculadora para estimar el grado de relación entre **actividades** y **objetivos específicos** del PEI.")

uploaded = st.file_uploader("Sube el Excel con las respuestas del *Formulario Único para el PEI*", type=["xlsx","xls"])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.subheader("Vista previa")
    st.dataframe(df.head(20))

    col_obj, col_act, col_uni, col_resp = guess_columns(df)

    with st.expander("Asignar columnas (editar si es necesario)"):
        cols = list(df.columns)
        col_obj = st.selectbox("Columna de **Objetivo específico**", cols, index=cols.index(col_obj) if col_obj in cols else 0)
        col_act = st.selectbox("Columna de **Actividad**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_uni = st.selectbox("Columna de **Unidad Académica** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_uni or col_uni not in cols else cols.index(col_uni)+1))
        col_resp = st.selectbox("Columna de **Responsable** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_resp or col_resp not in cols else cols.index(col_resp)+1))

        col_uni = None if col_uni == "(ninguna)" else col_uni
        col_resp = None if col_resp == "(ninguna)" else col_resp

    if st.button("Calcular consistencia"):
        out = evaluate(df, col_obj, col_act, col_uni, col_resp)
        by_unidad, by_obj = aggregations(out)

        st.success("¡Listo! Se calculó la consistencia.")
        st.subheader("Resultados por actividad")
        st.dataframe(out)

        st.subheader("Resumen por objetivo específico")
        st.dataframe(by_obj)

        if by_unidad is not None:
            st.subheader("Resumen por unidad académica")
            st.dataframe(by_unidad)

        # Excel (incluye todas las filas, marcando vacíos)
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Resultados_actividades")
            by_obj.to_excel(writer, index=False, sheet_name="Resumen_objetivos")
            if by_unidad is not None:
                by_unidad.to_excel(writer, index=False, sheet_name="Resumen_unidades")
        st.download_button("⬇️ Descargar resultados en Excel", data=buf_excel.getvalue(), file_name="consistencia_actividades_pei.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Promedio excluyendo "Sin objetivo (vacío)"
        mask_valid = out["Objetivo específico"] != "Sin objetivo (vacío)"
        n_excluidos = int((~mask_valid).sum())
        promedio_excl = float(out.loc[mask_valid, "consistencia_%"].mean()) if mask_valid.any() else 0.0

        # Word
        buf_word = generar_informe_word(promedio_excl, out, by_obj, by_unidad, n_excluidos=n_excluidos)
        st.download_button("⬇️ Descargar informe en Word", data=buf_word, file_name="informe_consistencia_pei.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # Métricas globales
        st.subheader("Indicadores globales")
        st.metric("Promedio general (excluye vacíos)", f"{promedio_excl:.1f}%")
        st.metric("Actividades evaluadas", f"{int(mask_valid.sum()):,}".replace(",", "."))
        if n_excluidos > 0:
            st.caption(f"Se excluyeron {n_excluidos} actividades con 'Objetivo específico' vacío (registradas como 'Sin objetivo (vacío)').")

else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")
