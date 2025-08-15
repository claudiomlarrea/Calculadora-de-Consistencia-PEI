import io
import re
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# ---------------- Utilidades ----------------
SPANISH_STOPWORDS = {"a","ante","bajo","cabe","con","contra","de","desde","durante","en","entre","hacia","hasta","mediante",
    "para","por","según","sin","so","sobre","tras","el","la","los","las","un","una","unos","unas","y","o","u","e","ni","que",
    "como","al","del","se","su","sus","es","son","ser","estar","esta","este","estos","estas","hay","más","menos","muy","ya",
    "no","sí","si","pero","porque","cuando","donde","cada","lo","le","les","también","además"}
BLANK_TOKENS = {"", "nan", "none", "s d", "sd", "s n d", "s n/d", "n a", "n/a", "no corresponde", "no aplica", "ninguno"}

def is_blank(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    return str(x).strip().lower() in BLANK_TOKENS

def to_clean_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)

def tokens_clean(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if t not in SPANISH_STOPWORDS]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def combined_score(activity: str, objective: str) -> float:
    t_ratio = fuzz.token_set_ratio(activity, objective) / 100.0
    jac = jaccard(tokens_clean(activity), tokens_clean(objective))
    return float(0.6 * t_ratio + 0.4 * jac)

# -------- Limpieza de “Objetivo” (evita mezclar con actividad/resultado) --------
code_pattern = re.compile(r"\b\d+(?:\.\d+)+\b")
def extract_goal_segment(text: str) -> str:
    s = str(text)
    parts = re.split(r"\s[-–—]\s", s)
    for part in reversed(parts):
        if code_pattern.search(part):
            return re.sub(r"\s*[-–—]\s*", " ", part.strip())
    m = re.search(r"(\d+(?:\.\d+)+\s+[^\n\r]+)$", s)
    return m.group(1).strip() if m else s.strip()

# -------- Detección de columnas --------
def best_column(df: pd.DataFrame, candidates) -> Optional[str]:
    best, best_score = None, -1
    for col in df.columns:
        col_norm = col.strip().lower()
        for cand in candidates:
            s = fuzz.partial_ratio(col_norm, cand.strip().lower())
            if s > best_score:
                best, best_score = col, s
    return best

def guess_columns(df: pd.DataFrame):
    col_obj_text = best_column(df, [
        "objetivo especifico", "objetivo específico", "objetivo", "objetivo pei", "objetivo del pei",
        "objetivo especifico al que tributa", "objetivo especifico (texto)"
    ])
    col_obj_code = best_column(df, [
        "codigo objetivo", "código objetivo", "id objetivo", "objetivo (codigo)","objetivo nro","objetivo numero"
    ])
    col_act = best_column(df, ["actividad", "acciones", "descripcion de la actividad", "descripción de la actividad", "actividad cargada"])
    col_act_alt = best_column(df, ["accion", "acción", "tarea", "detalle de actividad", "descripcion", "descripción"])
    col_group = best_column(df, ["unidad academica", "unidad académica", "facultad", "instituto", "secretaria", "secretaría"])
    return col_obj_text, col_obj_code, col_act, col_act_alt, col_group

# -------- Rellenos --------
def forward_fill_col(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[col] = df[col].replace("", np.nan)
        df[col] = df.groupby(group_cols, dropna=False)[col].ffill()
    else:
        df[col] = df[col].replace("", np.nan).ffill()
    return df.fillna({col: ""})

def back_fill_col(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    if group_cols:
        df[col] = df[col].replace("", np.nan)
        df[col] = df.groupby(group_cols, dropna=False)[col].bfill()
    else:
        df[col] = df[col].replace("", np.nan).bfill()
    return df.fillna({col: ""})

# -------- Evaluación (devuelve resultados + estadísticas de depuración) --------
def evaluate(df: pd.DataFrame, col_obj_text: str, col_obj_code: Optional[str],
             col_act: str, col_act_alt: Optional[str], group_col: Optional[str],
             use_ffill_goal: bool, use_bfill_goal: bool, use_ffill_act: bool, use_bfill_act: bool,
             use_fallback_activity: bool, combine_code_and_text: bool,
             clean_goal_text: bool, keep_empty_activity: bool=True, drop_duplicates: bool=False
             ) -> Tuple[pd.DataFrame, Dict[str,int]]:
    stats = {}
    stats["filas_formulario"] = len(df)

    work = df.copy()

    # Objetivo base
    objetivo_text = to_clean_str_series(work[col_obj_text])
    if clean_goal_text:
        objetivo_text = objetivo_text.apply(extract_goal_segment)

    # Combinar código + texto (opcional)
    if combine_code_and_text and col_obj_code and col_obj_code in work.columns:
        code_clean = to_clean_str_series(work[col_obj_code]).str.strip()
        have_code = ~code_clean.eq("")
        objetivo_text = objetivo_text.mask(have_code, code_clean + " - " + objetivo_text)

    # Actividad con fallback
    act_primary = to_clean_str_series(work[col_act])
    if use_fallback_activity and col_act_alt and col_act_alt in work.columns:
        act_alt = to_clean_str_series(work[col_act_alt])
        use_alt = act_primary.str.strip().eq("")
        act_primary = act_primary.mask(use_alt, act_alt)

    work["_objetivo"] = objetivo_text
    work["_actividad"] = act_primary
    if group_col and group_col in work.columns:
        work["_grupo"] = to_clean_str_series(work[group_col])
        groups = ["_grupo"]
    else:
        groups = []

    # Rellenos
    if use_ffill_goal: work = forward_fill_col(work, "_objetivo", groups)
    if use_bfill_goal: work = back_fill_col(work, "_objetivo", groups)
    if use_ffill_act:  work = forward_fill_col(work, "_actividad", groups)
    if use_bfill_act:  work = back_fill_col(work, "_actividad", groups)

    # Contar vacías antes de decidir mantener/eliminar
    empty_act_mask = work["_actividad"].apply(is_blank)
    stats["actividades_vacias_detectadas"] = int(empty_act_mask.sum())

    if keep_empty_activity:
        # marcamos como "(vacía)" y score 0
        work.loc[empty_act_mask, "_actividad"] = "(vacía)"
        work["_actividad_es_vacia"] = empty_act_mask
    else:
        work = work[~empty_act_mask].copy()

    # Objetivos vacíos -> etiqueta, pero no se descartan
    vacio_obj_mask = work["_objetivo"].apply(is_blank)
    stats["objetivos_vacios_detectados"] = int(vacio_obj_mask.sum())
    work.loc[vacio_obj_mask, "_objetivo"] = "Sin objetivo (vacío)"

    # Duplicados
    before_dedup = len(work)
    if drop_duplicates:
        work = work.drop_duplicates(subset=["_objetivo","_actividad"], keep="first")
    stats["duplicados_colapsados"] = before_dedup - len(work)

    # Score
    work["consistencia_%"] = work.apply(
        lambda r: 0.0 if ("_actividad_es_vacia" in work.columns and r.get("_actividad_es_vacia", False))
        else round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1),
        axis=1
    )

    out = work.rename(columns={"_objetivo":"Objetivo específico","_actividad":"Actividad"})[
        ["Objetivo específico","Actividad","consistencia_%"]
    ]
    stats["filas_informe"] = len(out)
    return out, stats

# ---------------- UI ----------------
st.set_page_config(page_title="Análisis de consistencia – Excel (v6e)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Excel (v6e)")
st.caption("Mantiene el conteo original (por defecto). Corrige objetivos mezclados, permite fallback de actividad y muestra diagnóstico del pipeline.")

uploaded = st.file_uploader("Sube el Excel del *Formulario Único para el PEI*", type=["xlsx","xls"])

df = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        sheet = xls.sheet_names[0] if len(xls.sheet_names)==1 else st.selectbox("Selecciona la hoja", xls.sheet_names, index=0)
        df = pd.read_excel(xls, sheet_name=sheet)
        st.caption(f"Hoja cargada: **{sheet}**")
    except Exception:
        df = pd.read_excel(uploaded)

if df is not None:
    st.subheader("Vista previa")
    st.dataframe(df.head(20))

    col_obj_text, col_obj_code, col_act, col_act_alt, col_group = guess_columns(df)

    with st.expander("Asignar columnas y opciones"):
        cols = list(df.columns)
        col_obj_text = st.selectbox("**Objetivo específico (texto)**", cols, index=cols.index(col_obj_text) if col_obj_text in cols else 0)
        col_obj_code = st.selectbox("**Código de objetivo** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_obj_code or col_obj_code not in cols else cols.index(col_obj_code)+1))
        col_act = st.selectbox("**Actividad (principal)**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_act_alt = st.selectbox("**Actividad (alternativa/fallback)**", ["(ninguna)"] + cols, index=(0 if not col_act_alt or col_act_alt not in cols else cols.index(col_act_alt)+1))
        col_group = st.selectbox("**Columna de agrupación (opcional)**", ["(ninguna)"] + cols, index=(0 if not col_group or col_group not in cols else cols.index(col_group)+1))

        col_obj_code = None if col_obj_code == "(ninguna)" else col_obj_code
        col_act_alt = None if col_act_alt == "(ninguna)" else col_act_alt
        col_group = None if col_group == "(ninguna)" else col_group

        combine_code_and_text = st.checkbox("Combinar **Código + Texto**", value=False)
        clean_goal_text = st.checkbox("Limpiar **Objetivo** (quedarse solo con '1.x …')", value=True)
        use_fallback_activity = st.checkbox("Usar **Actividad alternativa** si la principal está vacía", value=True)

        use_ffill_goal = st.checkbox("Rellenar **Objetivo** hacia abajo (forward-fill)", value=False)
        use_bfill_goal = st.checkbox("Rellenar **Objetivo** hacia arriba (backfill)", value=False)
        use_ffill_act = st.checkbox("Rellenar **Actividad** hacia abajo (forward-fill)", value=False)
        use_bfill_act = st.checkbox("Rellenar **Actividad** hacia arriba (backfill)", value=False)

        keep_empty_activity = st.checkbox("**Mantener actividades vacías** con 0% (conservar conteo)", value=True)
        drop_duplicates = st.checkbox("Eliminar **duplicados** (Objetivo, Actividad)", value=False)

        group_cols = [col_group] if col_group else []

    if st.button("Calcular y descargar Excel"):
        out, stats = evaluate(
            df, col_obj_text, col_obj_code, col_act, col_act_alt, col_group,
            use_ffill_goal, use_bfill_goal, use_ffill_act, use_bfill_act,
            use_fallback_activity, combine_code_and_text, clean_goal_text,
            keep_empty_activity=keep_empty_activity, drop_duplicates=drop_duplicates
        )

        promedio_global = round(float(out["consistencia_%"].mean()), 1) if len(out) else 0.0
        informe = pd.DataFrame({
            "Objetivo específico": out["Objetivo específico"],
            "Actividad específica cargada": out["Actividad"],
            "Porcentaje de correlación o consistencia de cada actividad": out["consistencia_%"],
            "Porcentaje de correlación total promedio": [promedio_global]*len(out)
        })

        st.success(f"¡Listo! {len(informe)} filas (promedio global: {promedio_global:.1f}%).")
        st.dataframe(informe.head(100))

        with st.expander("Diagnóstico del pipeline"):
            st.write({
                "Filas en el formulario": stats["filas_formulario"],
                "Actividades vacías detectadas": stats["actividades_vacias_detectadas"],
                "Objetivos vacíos detectados": stats["objetivos_vacios_detectados"],
                "Duplicados colapsados": stats["duplicados_colapsados"],
                "Filas en el informe final": stats["filas_informe"],
            })
            st.caption("Si el informe trae menos filas que el formulario, desmarca 'Eliminar duplicados' y/o activa el fallback/rellenos.")

        # Excel
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            informe.to_excel(writer, index=False, sheet_name="Informe")
        st.download_button("⬇️ Descargar Informe Excel", data=buf_excel.getvalue(),
                           file_name="informe_consistencia_pei.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")





