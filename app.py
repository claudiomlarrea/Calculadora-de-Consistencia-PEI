import io
import re
import unicodedata
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# ---------------------------
# Utilidades de texto
# ---------------------------
SPANISH_STOPWORDS = {
    "a","ante","bajo","cabe","con","contra","de","desde","durante","en","entre","hacia","hasta","mediante",
    "para","por","según","sin","so","sobre","tras","el","la","los","las","un","una","unos","unas","y","o",
    "u","e","ni","que","como","al","del","se","su","sus","es","son","ser","estar","esta","este","estos","estas",
    "hay","más","menos","muy","ya","no","sí","si","pero","porque","cuando","donde","entre","sobre","cada",
    "lo","le","les","debe","deben","deber","deberá","deberán","debería","deberían","puede","pueden","podrá",
    "podrán","podría","podrían","también","además"
}
BLANK_TOKENS = {"", "nan", "none", "s d", "sd", "s n d", "s n/d", "n a", "n/a", "no corresponde", "no aplica", "ninguno"}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s/]", " ", s)  # conservar "/" para reconocer n/a
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in normalize_text(s).split() if t and t not in SPANISH_STOPWORDS]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def combined_score(activity: str, objective: str) -> float:
    """
    Métrica de consistencia 0..1 combinando:
    - RapidFuzz token_set_ratio (60%)
    - Jaccard de tokens sin stopwords (40%)
    """
    t_ratio = fuzz.token_set_ratio(activity, objective) / 100.0
    jac = jaccard(tokens(activity), tokens(objective))
    return float(0.6 * t_ratio + 0.4 * jac)

def is_blank(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    s = normalize_text(str(x))
    return s in BLANK_TOKENS

def to_clean_str_series(s: pd.Series) -> pd.Series:
    # evita el string "nan": primero fillna(""), luego astype(str)
    return s.fillna("").astype(str)

# ---------------------------
# Detección de columnas
# ---------------------------
def best_column(df: pd.DataFrame, candidates) -> Optional[str]:
    best, best_score = None, -1
    for col in df.columns:
        col_norm = normalize_text(col)
        for cand in candidates:
            s = fuzz.partial_ratio(col_norm, normalize_text(cand))
            if s > best_score:
                best, best_score = col, s
    return best

def guess_columns(df: pd.DataFrame):
    col_obj_text = best_column(df, [
        "objetivo especifico", "objetivo", "objetivo pei", "objetivo del pei",
        "objetivo especifico del pei", "objetivo especifico al que tributa",
        "objetivo especifico seleccionado", "objetivo especifico (texto)"
    ])
    col_obj_code = best_column(df, [
        "codigo objetivo", "código objetivo", "id objetivo", "objetivo (codigo)",
        "objetivo n", "objetivo numero", "objetivo nro"
    ])
    col_act = best_column(df, ["actividad", "acciones", "descripcion de la actividad", "descripción de la actividad", "accion", "actividad prevista", "actividad cargada"])
    col_uni = best_column(df, ["unidad academica", "unidad académica", "unidad", "facultad", "instituto", "secretaria", "secretaría"])
    return col_obj_text, col_obj_code, col_act, col_uni

# ---------------------------
# Preparación de objetivo + limpieza
# ---------------------------
def make_objective(text_col: pd.Series, code_col: Optional[pd.Series]) -> pd.Series:
    text_clean = to_clean_str_series(text_col)
    if code_col is None:
        return text_clean
    code_clean = to_clean_str_series(code_col).str.strip()
    have_code = ~code_clean.eq("")
    combined = text_clean.copy()
    combined.loc[have_code] = code_clean.loc[have_code] + " - " + text_clean.loc[have_code]
    return combined

def forward_fill_objectives(df: pd.DataFrame, obj_col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[obj_col] = df[obj_col].replace("", np.nan)
        df[obj_col] = df.groupby(group_cols, dropna=False)[obj_col].ffill()
    else:
        df[obj_col] = df[obj_col].replace("", np.nan).ffill()
    df[obj_col] = df[obj_col].fillna("")
    return df

def back_fill_objectives(df: pd.DataFrame, obj_col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[obj_col] = df[obj_col].replace("", np.nan)
        df[obj_col] = df.groupby(group_cols, dropna=False)[obj_col].bfill()
    else:
        df[obj_col] = df[obj_col].replace("", np.nan).bfill()
    df[obj_col] = df[obj_col].fillna("")
    return df

# ---------------------------
# Evaluación y armado de informe
# ---------------------------
def evaluate(df: pd.DataFrame, col_obj_text: str, col_obj_code: Optional[str], col_act: str,
             col_uni: Optional[str], use_ffill: bool, use_bfill: bool, group_cols: list) -> pd.DataFrame:
    work = df.copy()

    # Objetivo: posible "Código - Texto"
    objetivo_series = make_objective(work[col_obj_text], work[col_obj_code] if col_obj_code else None)
    actividad_series = to_clean_str_series(work[col_act])

    work["_objetivo"] = objetivo_series
    work["_actividad"] = actividad_series
    if col_uni: work["_unidad"] = to_clean_str_series(work[col_uni])

    # Completar vacíos si corresponde
    group_cols_valid = [c for c in group_cols if c in work.columns]
    if use_ffill:
        work = forward_fill_objectives(work, "_objetivo", group_cols_valid)
    if use_bfill:
        work = back_fill_objectives(work, "_objetivo", group_cols_valid)

    # Etiquetar vacíos
    vacio_mask = work["_objetivo"].apply(is_blank)
    work.loc[vacio_mask, "_objetivo"] = "Sin objetivo (vacío)"

    # Score de consistencia
    work["consistencia_%"] = work.apply(lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1), axis=1)

    # Salida base
    out = work.rename(columns={
        "_objetivo":"Objetivo específico",
        "_actividad":"Actividad"
    })[["Objetivo específico","Actividad","consistencia_%"]]
    return out

# ---------------------------
# UI Streamlit
# ---------------------------
st.set_page_config(page_title="Análisis de consistencia de actividades PEI – Excel", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Excel")
st.caption("Genera un **único informe en Excel** con 4 columnas: Objetivo, Actividad, % de consistencia por actividad y **promedio global**.")

uploaded = st.file_uploader("Sube el Excel con las respuestas del *Formulario Único para el PEI*", type=["xlsx","xls"])

df = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("Selecciona la hoja del Excel", xls.sheet_names, index=0)
        else:
            sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
        st.caption(f"Hoja cargada: **{sheet}**")
    except Exception:
        df = pd.read_excel(uploaded)

if df is not None:
    st.subheader("Vista previa")
    st.dataframe(df.head(20))

    col_obj_text, col_obj_code, col_act, col_uni = guess_columns(df)

    with st.expander("Asignar columnas y opciones (editar si es necesario)"):
        cols = list(df.columns)
        col_obj_text = st.selectbox("Columna de **Objetivo específico (texto)**", cols, index=cols.index(col_obj_text) if col_obj_text in cols else 0)
        col_obj_code = st.selectbox("Columna de **Código de objetivo** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_obj_code or col_obj_code not in cols else cols.index(col_obj_code)+1))
        col_act = st.selectbox("Columna de **Actividad**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_uni = st.selectbox("Columna de **Unidad Académica** (opcional, útil para agrupar)", ["(ninguna)"] + cols, index=(0 if not col_uni or col_uni not in cols else cols.index(col_uni)+1))

        col_obj_code = None if col_obj_code == "(ninguna)" else col_obj_code
        col_uni = None if col_uni == "(ninguna)" else col_uni

        # Opciones de completado
        prev_obj = make_objective(df[col_obj_text], df[col_obj_code] if col_obj_code else None)
        vacios_prev = int(prev_obj.apply(lambda x: 1 if is_blank(x) else 0).sum())
        total_prev = int(len(prev_obj))
        suggest_ffill = (total_prev > 0 and (vacios_prev / total_prev) > 0.1)

        use_ffill = st.checkbox("Rellenar hacia abajo (forward-fill) los objetivos vacíos", value=suggest_ffill)
        use_bfill = st.checkbox("Rellenar hacia arriba (backfill) los objetivos vacíos", value=False)
        group_cols = st.multiselect(
            "Columnas para agrupar antes de completar (recomendado: Unidad Académica)",
            options=cols,
            default=[c for c in ["Unidad Académica", "unidad", "facultad"] if c in cols]
        )

    if st.button("Calcular y generar informe Excel"):
        out = evaluate(df, col_obj_text, col_obj_code, col_act, col_uni, use_ffill, use_bfill, group_cols)

        # Informe con 4 columnas exactas
        promedio_global = round(float(out["consistencia_%"].mean()), 1) if len(out) else 0.0
        informe = pd.DataFrame({
            "Objetivo específico": out["Objetivo específico"],
            "Actividad específica cargada": out["Actividad"],
            "Porcentaje de correlación o consistencia de cada actividad": out["consistencia_%"],
            "Porcentaje de correlación total promedio": [promedio_global]*len(out)
        })

        st.success("¡Listo! Informe preparado.")
        st.dataframe(informe)

        # Exportar a Excel (una sola hoja)
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            informe.to_excel(writer, index=False, sheet_name="Informe")
        st.download_button("⬇️ Descargar Informe Excel", data=buf_excel.getvalue(),
                           file_name="informe_consistencia_pei.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Indicador global
        st.subheader("Indicador global")
        st.metric("Porcentaje de correlación total promedio", f"{promedio_global:.1f}%")
else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")

    st.info("Sube el archivo Excel de respuestas para comenzar.")


