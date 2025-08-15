import io
import re
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from docx import Document

# ---------------- Utilidades ----------------
SPANISH_STOPWORDS = {
    "a","ante","bajo","cabe","con","contra","de","desde","durante","en","entre","hacia","hasta","mediante",
    "para","por","según","sin","so","sobre","tras","el","la","los","las","un","una","unos","unas","y","o","u","e","ni","que",
    "como","al","del","se","su","sus","es","son","ser","estar","esta","este","estos","estas","hay","más","menos","muy","ya",
    "no","sí","si","pero","porque","cuando","donde","cada","lo","le","les","también","además"
}
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

# -------- Limpieza estricta de “Objetivo específico” --------
CODE_RE = re.compile(r"\d+(?:\.\d+)+")  # p.ej. 1.4, 1.5.2

def force_goal_only(text: str) -> str:
    """
    Devuelve exclusivamente el tramo '1.x ...' del objetivo.
    - Busca el primer match del patrón 1.x en cualquier parte del texto.
    - Devuelve desde ese match hasta el final de la línea (sin prefijos de actividad/resultado).
    - Si no hay código, devuelve el texto original (ya limpio).
    """
    s = str(text).strip()
    m = CODE_RE.search(s)
    if not m:
        return s
    start = m.start()
    # desde el código hasta el final; además, si hay separadores ' - ', cortamos lo previo
    tail = s[start:]
    # quitar separadores repetidos
    tail = re.sub(r"\s*[-–—]\s*", " ", tail).strip()
    return tail

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
        "objetivo especifico", "objetivo específico", "objetivo", "objetivo pei",
        "objetivo del pei", "objetivo especifico al que tributa", "objetivo especifico (texto)"
    ])
    col_obj_code = best_column(df, [
        "codigo objetivo", "código objetivo", "id objetivo", "objetivo (codigo)","objetivo nro","objetivo numero"
    ])
    col_act = best_column(df, [
        "actividad", "acciones", "descripcion de la actividad", "descripción de la actividad", "actividad cargada"
    ])
    col_act_alt = best_column(df, [
        "accion", "acción", "tarea", "detalle de actividad", "descripcion", "descripción"
    ])
    col_group = best_column(df, [
        "unidad academica", "unidad académica", "facultad", "instituto", "secretaria", "secretaría"
    ])
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

# -------- Evaluación --------
def evaluate(
    df: pd.DataFrame,
    col_obj_text: str, col_obj_code: Optional[str],
    col_act: str, col_act_alt: Optional[str], group_col: Optional[str],
    use_ffill_goal: bool, use_bfill_goal: bool, use_ffill_act: bool, use_bfill_act: bool,
    use_fallback_activity: bool, combine_code_and_text: bool
) -> pd.DataFrame:

    work = df.copy()

    # Objetivo: SIEMPRE limpiar a '1.x …'
    objetivo_text = to_clean_str_series(work[col_obj_text]).apply(force_goal_only)

    # NO combinar código + texto por defecto; solo si el usuario lo pide
    if combine_code_and_text and col_obj_code and col_obj_code in work.columns:
        code_clean = to_clean_str_series(work[col_obj_code]).str.strip()
        have_code = ~code_clean.eq("")
        objetivo_text = objetivo_text.mask(have_code, code_clean + " " + objetivo_text)

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

    # Marcar objetivos vacíos y ELIMINARLOS (requisito)
    vacio_obj_mask = work["_objetivo"].apply(is_blank)
    work.loc[vacio_obj_mask, "_objetivo"] = "Sin objetivo (vacío)"
    work = work[work["_objetivo"] != "Sin objetivo (vacío)"].copy()

    # Consistencia
    work["consistencia_%"] = work.apply(
        lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1),
        axis=1
    )

    return work.rename(columns={"_objetivo":"Objetivo específico","_actividad":"Actividad"})[
        ["Objetivo específico","Actividad","consistencia_%"]
    ]

# -------- Informe Word --------
def generar_informe_word(n_acts: int, promedio: float, dist: Dict[str,int]) -> bytes:
    doc = Document()
    doc.add_heading("Conclusiones de Consistencia de actividades", 0)

    p = doc.add_paragraph()
    p.add_run("Cantidad de actividades evaluadas: ").bold = True
    p.add_run(str(n_acts))

    p = doc.add_paragraph()
    p.add_run("Porcentaje promedio de consistencia general: ").bold = True
    p.add_run(f"{promedio:.1f}%")

    doc.add_heading("Interpretación de los resultados", level=1)
    if promedio >= 75:
        doc.add_paragraph(
            "El promedio global indica una consistencia alta entre actividades y objetivos específicos. "
            "Las descripciones de actividades, en general, reflejan de manera clara los verbos, ámbitos y productos esperados por los objetivos del PEI. "
            "Se recomienda consolidar buenas prácticas y estandarizar plantillas de redacción."
        )
    elif promedio >= 50:
        doc.add_paragraph(
            "El promedio global ubica la consistencia en un nivel intermedio. "
            "Existen tramos con buena alineación y otros con desajustes (actividades genéricas o productos poco definidos). "
            "Conviene revisar los objetivos con menor consistencia y ajustar los criterios de vinculación."
        )
    else:
        doc.add_paragraph(
            "La consistencia global es baja; hay señales de desalineación entre lo que se ejecuta y lo que plantean los objetivos específicos. "
            "Se aconseja reescribir actividades para que expresen explícitamente el aporte al objetivo (verbo de acción, ámbito/población, entregable y resultado esperado)."
        )

    doc.add_heading("Distribución por niveles", level=2)
    t = doc.add_table(rows=1, cols=2)
    t.style = "Light List Accent 1"
    t.cell(0,0).text = "Nivel"
    t.cell(0,1).text = "N actividades"
    for k in ["Alta (>=75%)","Media (50–74%)","Baja (<50%)"]:
        row = t.add_row().cells
        row[0].text = k
        row[1].text = str(dist.get(k,0))

    doc.add_heading("Recomendaciones", level=1)
    for r in [
        "Usar verbos operativos y objeto claro en la redacción de actividades.",
        "Mencionar el entregable/resultado y el ámbito/población objetivo.",
        "Evitar actividades duplicadas o genéricas; agruparlas como líneas con sub-tareas medibles.",
        "Revisar los objetivos con mayor proporción en nivel 'Baja' para realinear la cartera."
    ]:
        doc.add_paragraph("• " + r)

    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    return buf.getvalue()

# ---------------- UI ----------------
st.set_page_config(page_title="Análisis de consistencia – Excel+Word (v7.1)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Excel + Word (v7.1)")
st.caption("Objetivo específico limpio (solo '1.x …'), exclusión de 'Sin objetivo (vacío)', Excel con 4 columnas e informe Word.")

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
        col_obj_code = st.selectbox("**Código de objetivo** (opcional, no se mezcla por defecto)", ["(ninguna)"] + cols, index=(0 if not col_obj_code or col_obj_code not in cols else cols.index(col_obj_code)+1))
        col_act = st.selectbox("**Actividad (principal)**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_act_alt = st.selectbox("**Actividad (alternativa/fallback)**", ["(ninguna)"] + cols, index=(0 if not col_act_alt or col_act_alt not in cols else cols.index(col_act_alt)+1))
        col_group = st.selectbox("**Columna de agrupación (opcional)**", ["(ninguna)"] + cols, index=(0 if not col_group or col_group not in cols else cols.index(col_group)+1))

        col_obj_code = None if col_obj_code == "(ninguna)" else col_obj_code
        col_act_alt = None if col_act_alt == "(ninguna)" else col_act_alt
        col_group = None if col_group == "(ninguna)" else col_group
        combine_code_and_text = st.checkbox("Combinar **Código + Objetivo (texto)**", value=False)

        use_ffill_goal = st.checkbox("Rellenar **Objetivo** hacia abajo (forward-fill)", value=False)
        use_bfill_goal = st.checkbox("Rellenar **Objetivo** hacia arriba (backfill)", value=False)
        use_ffill_act = st.checkbox("Rellenar **Actividad** hacia abajo (forward-fill)", value=False)
        use_bfill_act = st.checkbox("Rellenar **Actividad** hacia arriba (backfill)", value=False)

        group_cols = [col_group] if col_group else []

    if st.button("Calcular y descargar Excel + Word"):
        out = evaluate(
            df, col_obj_text, col_obj_code, col_act, col_act_alt, col_group,
            use_ffill_goal, use_bfill_goal, use_ffill_act, use_bfill_act,
            use_fallback_activity=True, combine_code_and_text=combine_code_and_text
        )

        promedio_global = round(float(out["consistencia_%"].mean()), 1) if len(out) else 0.0
        informe = pd.DataFrame({
            "Objetivo específico": out["Objetivo específico"],
            "Actividad específica cargada": out["Actividad"],
            "Porcentaje de correlación o consistencia de cada actividad": out["consistencia_%"],
            "Porcentaje de correlación total promedio": [promedio_global]*len(out)
        })

        st.success(f"¡Listo! {len(informe)} actividades evaluadas (promedio: {promedio_global:.1f}%).")
        st.dataframe(informe.head(100))

        # Excel
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            informe.to_excel(writer, index=False, sheet_name="Informe")
        st.download_button("⬇️ Descargar Informe Excel", data=buf_excel.getvalue(),
                           file_name="informe_consistencia_pei.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Word
        alta = int((out["consistencia_%"] >= 75).sum())
        media = int(((out["consistencia_%"] >= 50) & (out["consistencia_%"] < 75)).sum())
        baja = int((out["consistencia_%"] < 50).sum())
        dist = {"Alta (>=75%)": alta, "Media (50–74%)": media, "Baja (<50%)": baja}

        word_bytes = generar_informe_word(n_acts=len(out), promedio=promedio_global, dist=dist)
        st.download_button("⬇️ Descargar Informe Word (Conclusiones)", data=word_bytes,
                           file_name="conclusiones_consistencia_actividades.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")






