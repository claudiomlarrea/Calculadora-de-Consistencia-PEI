import io
import re
import unicodedata
from typing import Optional, List
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
    s = re.sub(r"[^a-z0-9\s/]", " ", s)
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
    t_ratio = fuzz.token_set_ratio(activity, objective) / 100.0
    jac = jaccard(tokens(activity), tokens(objective))
    return float(0.6 * t_ratio + 0.4 * jac)

def is_blank(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    s = normalize_text(str(x))
    return s in BLANK_TOKENS

def to_clean_str_series(s: pd.Series) -> pd.Series:
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
    col_act_alt = best_column(df, ["accion", "acción", "tarea", "detalle actividad", "detalle de actividad", "descripcion", "descripción"])
    col_uni = best_column(df, ["unidad academica", "unidad académica", "unidad", "facultad", "instituto", "secretaria", "secretaría"])
    return col_obj_text, col_obj_code, col_act, col_act_alt, col_uni

# ---------------------------
# Preparación de columnas
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

def combine_activity(primary: pd.Series, alternate: Optional[pd.Series]) -> pd.Series:
    a = to_clean_str_series(primary)
    if alternate is None:
        return a
    b = to_clean_str_series(alternate)
    use_b = a.str.strip().eq("")
    a.loc[use_b] = b.loc[use_b]
    return a

def forward_fill_col(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[col] = df[col].replace("", np.nan)
        df[col] = df.groupby(group_cols, dropna=False)[col].ffill()
    else:
        df[col] = df[col].replace("", np.nan).ffill()
    df[col] = df[col].fillna("")
    return df

def back_fill_col(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[col] = df[col].replace("", np.nan)
        df[col] = df.groupby(group_cols, dropna=False)[col].bfill()
    else:
        df[col] = df[col].replace("", np.nan).bfill()
    df[col] = df[col].fillna("")
    return df

# ---------------------------
# Evaluación
# ---------------------------
def evaluate(df: pd.DataFrame, col_obj_text: str, col_obj_code: Optional[str],
             col_act: str, col_act_alt: Optional[str], col_uni: Optional[str],
             use_ffill_obj: bool, use_bfill_obj: bool, use_ffill_act: bool, use_bfill_act: bool,
             group_cols: list, drop_empty_activity: bool=True, drop_duplicates: bool=True) -> pd.DataFrame:
    work = df.copy()

    # Objetivo: posible "Código - Texto"
    objetivo_series = make_objective(work[col_obj_text], work[col_obj_code] if col_obj_code else None)
    # Actividad: primaria + fallback
    actividad_series = combine_activity(work[col_act], work[col_act_alt] if col_act_alt else None)

    work["_objetivo"] = to_clean_str_series(objetivo_series)
    work["_actividad"] = to_clean_str_series(actividad_series)
    if col_uni: work["_unidad"] = to_clean_str_series(work[col_uni])

    group_cols_valid = [c for c in group_cols if c in work.columns]

    # Completar OBJETIVOS si corresponde
    if use_ffill_obj:
        work = forward_fill_col(work, "_objetivo", group_cols_valid)
    if use_bfill_obj:
        work = back_fill_col(work, "_objetivo", group_cols_valid)

    # Completar ACTIVIDADES si corresponde
    if use_ffill_act:
        work = forward_fill_col(work, "_actividad", group_cols_valid)
    if use_bfill_act:
        work = back_fill_col(work, "_actividad", group_cols_valid)

    if drop_empty_activity:
        work = work[~work["_actividad"].apply(is_blank)].copy()

    # Etiquetar objetivos vacíos
    vacio_mask = work["_objetivo"].apply(is_blank)
    work.loc[vacio_mask, "_objetivo"] = "Sin objetivo (vacío)"

    # Eliminar duplicados (Objetivo, Actividad)
    if drop_duplicates:
        work = work.drop_duplicates(subset=["_objetivo","_actividad"], keep="first")

    # Score de consistencia
    work["consistencia_%"] = work.apply(lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1), axis=1)

    out = work.rename(columns={
        "_objetivo":"Objetivo específico",
        "_actividad":"Actividad"
    })[["Objetivo específico","Actividad","consistencia_%"]]
    return out

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Análisis de consistencia de actividades PEI – Excel (v6c)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Excel (v6c)")
st.caption("Exporta un único Excel con 4 columnas limpias. Incluye fallback de actividad, completar hacia arriba/abajo y deduplicación.")

uploaded = st.file_uploader("Sube el Excel del *Formulario Único para el PEI*", type=["xlsx","xls"])

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

    col_obj_text, col_obj_code, col_act, col_act_alt, col_uni = guess_columns(df)

    with st.expander("Asignar columnas y opciones"):
        cols = list(df.columns)
        col_obj_text = st.selectbox("**Objetivo específico (texto)**", cols, index=cols.index(col_obj_text) if col_obj_text in cols else 0)
        col_obj_code = st.selectbox("**Código de objetivo** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_obj_code or col_obj_code not in cols else cols.index(col_obj_code)+1))
        col_act = st.selectbox("**Actividad (principal)**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_act_alt = st.selectbox("**Actividad (alternativa/fallback)**", ["(ninguna)"] + cols, index=(0 if not col_act_alt or col_act_alt not in cols else cols.index(col_act_alt)+1))
        col_uni = st.selectbox("**Columna de agrupación** (recomendado: Unidad Académica)", ["(ninguna)"] + cols, index=(0 if not col_uni or col_uni not in cols else cols.index(col_uni)+1))

        col_obj_code = None if col_obj_code == "(ninguna)" else col_obj_code
        col_act_alt = None if col_act_alt == "(ninguna)" else col_act_alt
        col_uni = None if col_uni == "(ninguna)" else col_uni

        # Estimar vacíos
        prev_obj = make_objective(df[col_obj_text], df[col_obj_code] if col_obj_code else None)
        prev_act = combine_activity(df[col_act], df[col_act_alt] if col_act_alt else None)
        prop_obj_vacios = (prev_obj == "").mean() if len(prev_obj) else 0
        prop_act_vacios = (prev_act == "").mean() if len(prev_act) else 0

        st.caption(f"Vacíos detectados — Objetivo: {prop_obj_vacios:.0%} | Actividad: {prop_act_vacios:.0%}")

        use_ffill_obj = st.checkbox("Rellenar OBJETIVOS hacia abajo (forward-fill)", value=prop_obj_vacios > 0.1)
        use_bfill_obj = st.checkbox("Rellenar OBJETIVOS hacia arriba (backfill)", value=False)
        use_ffill_act = st.checkbox("Rellenar ACTIVIDADES hacia abajo (forward-fill)", value=prop_act_vacios > 0.1)
        use_bfill_act = st.checkbox("Rellenar ACTIVIDADES hacia arriba (backfill)", value=False)

        group_cols = st.multiselect("Agrupar por antes de completar (evita mezclar bloques)", options=cols, default=[c for c in ["Unidad Académica","unidad","facultad"] if c in cols])

        drop_empty_activity = st.checkbox("Eliminar filas con **Actividad** vacía", value=True)
        drop_duplicates = st.checkbox("Eliminar **duplicados** (Objetivo, Actividad)", value=True)

    if st.button("Calcular y descargar Excel"):
        out = evaluate(
            df, col_obj_text, col_obj_code, col_act, col_act_alt, col_uni,
            use_ffill_obj, use_bfill_obj, use_ffill_act, use_bfill_act,
            group_cols, drop_empty_activity=drop_empty_activity, drop_duplicates=drop_duplicates
        )

        # Informe con 4 columnas exactas
        promedio_global = round(float(out["consistencia_%"].mean()), 1) if len(out) else 0.0
        informe = pd.DataFrame({
            "Objetivo específico": out["Objetivo específico"],
            "Actividad específica cargada": out["Actividad"],
            "Porcentaje de correlación o consistencia de cada actividad": out["consistencia_%"],
            "Porcentaje de correlación total promedio": [promedio_global]*len(out)
        })

        st.success(f"¡Listo! {len(informe)} filas en el informe (promedio global: {promedio_global:.1f}%).")
        st.dataframe(informe.head(100))

        # Excel
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            informe.to_excel(writer, index=False, sheet_name="Informe")
        st.download_button(
            "⬇️ Descargar Informe Excel",
            data=buf_excel.getvalue(),
            file_name="informe_consistencia_pei.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")



