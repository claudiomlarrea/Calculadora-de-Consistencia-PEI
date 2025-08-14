
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

BLANK_TOKENS = {"", "nan", "none", "s d", "sd", "s n d", "s n/d", "n a", "n/a", "no corresponde", "no aplica", "ninguno"}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s/]", " ", s)  # mantener "/" para reconocer "n/a"
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
    col_obj_text = best_column(df, [
        "objetivo especifico", "objetivo", "objetivo pei", "objetivo del pei",
        "objetivo especifico del pei", "objetivo especifico al que tributa",
        "objetivo especifico seleccionado", "objetivo especifico (texto)"
    ])
    col_obj_code = best_column(df, [
        "codigo objetivo", "código objetivo", "id objetivo", "objetivo (codigo)",
        "objetivo n", "objetivo numero", "objetivo nro"
    ])
    col_act = best_column(df, ["actividad", "acciones", "descripcion de la actividad", "accion", "actividad prevista", "actividad cargada"])
    col_uni = best_column(df, ["unidad academica", "unidad", "facultad", "instituto", "secretaria"])
    col_resp = best_column(df, ["responsable", "responsables", "area responsable"])
    return col_obj_text, col_obj_code, col_act, col_uni, col_resp

def is_blank(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    s = normalize_text(str(x))
    return s in BLANK_TOKENS

def to_clean_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)

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

def evaluate(df: pd.DataFrame, col_obj_text: str, col_obj_code: Optional[str], col_act: str, col_uni: Optional[str], col_resp: Optional[str],
             use_ffill: bool, use_bfill: bool, group_cols: list) -> pd.DataFrame:
    work = df.copy()

    # Construir objetivo (posible código + texto)
    objetivo_series = make_objective(work[col_obj_text], work[col_obj_code] if col_obj_code else None)
    actividad_series = to_clean_str_series(work[col_act])

    work["_objetivo"] = objetivo_series
    work["_actividad"] = actividad_series
    if col_uni: work["_unidad"] = to_clean_str_series(work[col_uni])
    if col_resp: work["_responsable"] = to_clean_str_series(work[col_resp])

    # Completar vacíos
    group_cols_valid = [c for c in group_cols if c in work.columns]
    if use_ffill:
        work = forward_fill_objectives(work, "_objetivo", group_cols_valid)
    if use_bfill:
        work = back_fill_objectives(work, "_objetivo", group_cols_valid)

    # Marcar vacíos
    vacio_mask = work["_objetivo"].apply(is_blank)
    work.loc[vacio_mask, "_objetivo"] = "Sin objetivo (vacío)"

    # Score
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

def best_suggestion(activity: str, current_obj: str, candidates: List[str]) -> Tuple[str, float]:
    # Devuelve (objetivo_sugerido, score%) evitando "Sin objetivo" y priorizando el mejor distinto al actual
    best_obj, best_sc = "", -1.0
    second_obj, second_sc = "", -1.0
    for obj in candidates:
        sc = combined_score(activity, obj) * 100.0
        if sc > best_sc:
            second_obj, second_sc = best_obj, best_sc
            best_obj, best_sc = obj, sc
        elif sc > second_sc:
            second_obj, second_sc = obj, sc
    # Si el mejor es el mismo que el actual, usar el segundo
    if best_obj == current_obj and second_obj:
        return second_obj, second_sc
    return best_obj, best_sc

def build_inconsistencies(out: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # Filtra "inconsistentes" por debajo del umbral y sugiere objetivo
    df = out.copy()
    df["is_inconsistente"] = df["consistencia_%"] < threshold
    inc = df[df["is_inconsistente"]].copy()
    # Candidatos de objetivos
    candidates = sorted([o for o in df["Objetivo específico"].dropna().unique().tolist() if o != "Sin objetivo (vacío)"])
    if not candidates:
        inc["Objetivo sugerido"] = ""
        inc["Similitud sugerida %"] = np.nan
        return inc
    sugeridos, sim_scores = [], []
    for _, r in inc.iterrows():
        sug, sc = best_suggestion(r["Actividad"], r["Objetivo específico"], candidates)
        sugeridos.append(sug)
        sim_scores.append(round(sc, 1))
    inc["Objetivo sugerido"] = sugeridos
    inc["Similitud sugerida %"] = sim_scores
    return inc

def generar_informe_word(promedio_excl, out_df, resumen_obj, resumen_uni=None, n_excluidos=0, ffill_info="", inconsistentes: Optional[pd.DataFrame]=None, threshold: float=50.0):
    doc = Document()
    doc.add_heading("Análisis de consistencia de actividades PEI", 0)
    doc.add_paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y')}")
    doc.add_paragraph(f"Promedio general de consistencia (excluye objetivos vacíos): {promedio_excl:.1f}%")
    if ffill_info:
        doc.add_paragraph(ffill_info)
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

    # Sección de inconsistencias
    if inconsistentes is not None and len(inconsistentes) > 0:
        doc.add_heading(f"Actividades inconsistentes (umbral {threshold:.0f}%) y sugerencia de vinculación", level=1)
        cols = ["Unidad Académica","Actividad","Objetivo específico","consistencia_%","Objetivo sugerido","Similitud sugerida %"]
        cols_presentes = [c for c in cols if c in inconsistentes.columns]
        table = doc.add_table(rows=1, cols=len(cols_presentes))
        table.style = "Light List Accent 2"
        for j, col in enumerate(cols_presentes):
            table.cell(0, j).text = col
        for _, row in inconsistentes.iterrows():
            cells = table.add_row().cells
            for j, col in enumerate(cols_presentes):
                text = str(row[col])
                # recortar actividad larga para mantener manejable el Word
                if col == "Actividad" and len(text) > 300:
                    text = text[:300] + "..."
                cells[j].text = text

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

df = None
if uploaded is not None:
    # Selector de hoja si hay varias
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

    col_obj_text, col_obj_code, col_act, col_uni, col_resp = guess_columns(df)

    with st.expander("Asignar columnas y opciones (editar si es necesario)"):
        cols = list(df.columns)
        col_obj_text = st.selectbox("Columna de **Objetivo específico (texto)**", cols, index=cols.index(col_obj_text) if col_obj_text in cols else 0)
        col_obj_code = st.selectbox("Columna de **Código de objetivo** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_obj_code or col_obj_code not in cols else cols.index(col_obj_code)+1))
        col_act = st.selectbox("Columna de **Actividad**", cols, index=cols.index(col_act) if col_act in cols else 0)
        col_uni = st.selectbox("Columna de **Unidad Académica** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_uni or col_uni not in cols else cols.index(col_uni)+1))
        col_resp = st.selectbox("Columna de **Responsable** (opcional)", ["(ninguna)"] + cols, index=(0 if not col_resp or col_resp not in cols else cols.index(col_resp)+1))

        col_obj_code = None if col_obj_code == "(ninguna)" else col_obj_code
        col_uni = None if col_uni == "(ninguna)" else col_uni
        col_resp = None if col_resp == "(ninguna)" else col_resp

        # Opciones de completar vacíos
        prev_obj = make_objective(df[col_obj_text], df[col_obj_code] if col_obj_code else None)
        vacios_prev = int(prev_obj.apply(lambda x: 1 if is_blank(x) else 0).sum())
        total_prev = int(len(prev_obj))
        suggest_ffill = (total_prev > 0 and (vacios_prev / total_prev) > 0.1)
        use_ffill = st.checkbox("Rellenar hacia abajo (forward-fill) los objetivos vacíos", value=suggest_ffill)
        use_bfill = st.checkbox("Rellenar hacia arriba (backfill) los objetivos vacíos", value=False)
        group_cols = st.multiselect(
            "Columnas para agrupar antes de completar (p. ej., Unidad Académica)",
            options=cols,
            default=[c for c in ["Unidad Académica", "unidad", "facultad"] if c in cols]
        )

        threshold = st.slider("Umbral para marcar 'inconsistente' (%)", min_value=0, max_value=100, value=50, step=1)

    if st.button("Calcular consistencia"):
        out = evaluate(df, col_obj_text, col_obj_code, col_act, col_uni, col_resp, use_ffill, use_bfill, group_cols)
        by_unidad, by_obj = aggregations(out)

        # Diagnóstico
        vacios = int((out["Objetivo específico"] == "Sin objetivo (vacío)").sum())
        total = int(len(out))
        with st.expander("Diagnóstico de objetivos vacíos / valores únicos"):
            st.write(f"Filas totales: {total} | 'Sin objetivo (vacío)': {vacios}")
            freq = out["Objetivo específico"].value_counts(dropna=False).reset_index()
            freq.columns = ["Objetivo", "Frecuencia"]
            st.dataframe(freq.head(50))

        if total > 0 and vacios / total > 0.05:
            st.warning(f"Quedaron {vacios} de {total} actividades ({vacios/total:.0%}) sin objetivo. Revisa columnas y/o activa forward/backfill por grupos.")

        st.success("¡Listo! Se calculó la consistencia.")
        st.subheader("Resultados por actividad")
        st.dataframe(out)

        st.subheader("Resumen por objetivo específico")
        st.dataframe(by_obj)

        if by_unidad is not None:
            st.subheader("Resumen por unidad académica")
            st.dataframe(by_unidad)

        # Excel
        buf_excel = io.BytesIO()
        with pd.ExcelWriter(buf_excel, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Resultados_actividades")
            by_obj.to_excel(writer, index=False, sheet_name="Resumen_objetivos")
            if by_unidad is not None:
                by_unidad.to_excel(writer, index=False, sheet_name="Resumen_unidades")
        st.download_button("⬇️ Descargar resultados en Excel", data=buf_excel.getvalue(), file_name="consistencia_actividades_pei.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Promedio excluyendo vacíos
        mask_valid = out["Objetivo específico"] != "Sin objetivo (vacío)"
        n_excluidos = int((~mask_valid).sum())
        promedio_excl = float(out.loc[mask_valid, "consistencia_%"].mean()) if mask_valid.any() else 0.0

        # Inconsistencias + sugerencias
        inconsistentes = build_inconsistencies(out[mask_valid], threshold=threshold)

        # Word
        ffill_info = ""
        if use_ffill or use_bfill:
            modo = []
            if use_ffill: modo.append("forward-fill")
            if use_bfill: modo.append("backfill")
            modo_txt = " y ".join(modo)
            if group_cols:
                ffill_info = f"Se aplicó {modo_txt} de objetivos vacíos agrupando por: {', '.join(group_cols)}."
            else:
                ffill_info = f"Se aplicó {modo_txt} de objetivos vacíos sin agrupación."

        buf_word = generar_informe_word(
            promedio_excl, out, by_obj, by_unidad,
            n_excluidos=n_excluidos, ffill_info=ffill_info,
            inconsistentes=inconsistentes, threshold=threshold
        )
        st.download_button("⬇️ Descargar informe en Word", data=buf_word, file_name="informe_consistencia_pei.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # Métricas globales
        st.subheader("Indicadores globales")
        st.metric("Promedio general (excluye vacíos)", f"{promedio_excl:.1f}%")
        st.metric("Actividades evaluadas", f"{int(mask_valid.sum()):,}".replace(",", "."))
        st.metric("Inconsistentes (por debajo del umbral)", f"{len(inconsistentes):,}".replace(",", "."))
        if n_excluidos > 0:
            st.caption(f"Se excluyeron {n_excluidos} actividades con 'Objetivo específico' vacío (registradas como 'Sin objetivo (vacío)').")

else:
    st.info("Sube el archivo Excel de respuestas para comenzar.")

