import io
import re
import unicodedata
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from docx import Document

# ===========================
# Utilidades y configuración
# ===========================
SPANISH_STOPWORDS = {
    "a","ante","bajo","cabe","con","contra","de","desde","durante","en","entre","hacia","hasta","mediante",
    "para","por","según","sin","so","sobre","tras","el","la","los","las","un","una","unos","unas","y","o","u","e","ni","que",
    "como","al","del","se","su","sus","es","son","ser","estar","esta","este","estos","estas","hay","más","menos","muy","ya",
    "no","sí","si","pero","porque","cuando","donde","cada","lo","le","les","también","además"
}
BLANK_TOKENS = {
    "", "nan", "none", "s d", "sd", "s n d", "s n/d", "n a", "n/a",
    "no corresponde", "no aplica", "ninguno", "0", "-", "–", "—", "✓"
}
CODE_RE = re.compile(r"\d+(?:\.\d+)+")  # ej. 1.4, 1.5.2

def is_blank(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    return str(x).strip().lower() in BLANK_TOKENS

def to_clean_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

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

def force_goal_only(text: str) -> str:
    """
    Devuelve exclusivamente el tramo '1.x …' del objetivo.
    Si no hay código, devuelve el texto original.
    """
    s = str(text).strip()
    m = CODE_RE.search(s)
    if not m:
        return s
    tail = s[m.start():]
    tail = re.sub(r"\s*[-–—]\s*", " ", tail).strip()
    return tail

def parse_top_objective_from_name(name: str) -> Optional[str]:
    m = re.search(r"[Oo]bjetivo\s*(\d+)", name)
    return f"Objetivo {m.group(1)}" if m else None

# Duplicados: normalización simple
def normalize_for_dupes(s: str) -> str:
    s = strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===========================
# Detección de columnas
# ===========================
def best_column(df: pd.DataFrame, candidates) -> Optional[str]:
    best, best_score = None, -1
    for col in df.columns:
        col_norm = col.strip().lower()
        for cand in candidates:
            s = fuzz.partial_ratio(col_norm, cand.strip().lower())
            if s > best_score:
                best, best_score = col, s
    return best

def guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    col_obj = best_column(df, [
        "objetivo especifico", "objetivo específico", "objetivos especificos", "objetivos específicos",
        "objetivo", "objetivo pei", "objetivo del pei"
    ])
    col_act = best_column(df, [
        "actividad", "actividades", "acciones", "actividad específica", "actividad especifica",
        "descripcion de la actividad", "descripción de la actividad"
    ])
    return col_obj, col_act

# ===========================
# Carga de archivos (CSV/XLSX)
# ===========================
def load_frames_from_upload(uploaded_file) -> List[pd.DataFrame]:
    """Devuelve una lista de DataFrames (una por archivo/hoja útil)."""
    frames = []
    name = getattr(uploaded_file, "name", "archivo")
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            frames.append(df)
        else:
            xls = pd.ExcelFile(uploaded_file)
            # Elegir la hoja "más prometedora"
            best_sheet, best_hit = None, -1
            for sh in xls.sheet_names:
                tmp = pd.read_excel(xls, sheet_name=sh)
                col_obj, col_act = guess_columns(tmp)
                hit = int(col_obj is not None) + int(col_act is not None)
                if hit > best_hit:
                    best_hit, best_sheet = hit, sh
            if best_sheet is None:
                best_sheet = xls.sheet_names[0]
            frames.append(pd.read_excel(xls, sheet_name=best_sheet))
    except Exception:
        frames.append(pd.read_excel(uploaded_file))
    return frames

# ===========================
# Evaluación por DataFrame
# ===========================
def evaluate_df(df: pd.DataFrame, top_obj_label: Optional[str]) -> pd.DataFrame:
    col_obj, col_act = guess_columns(df)
    if not col_obj or not col_act:
        return pd.DataFrame(columns=[
            "Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)"
        ])

    objetivo = to_clean_str_series(df[col_obj]).apply(force_goal_only)
    actividad = to_clean_str_series(df[col_act])

    work = pd.DataFrame({
        "_objetivo": objetivo,
        "_actividad": actividad
    })

    # Excluir objetivos vacíos
    vacio_obj = work["_objetivo"].apply(is_blank)
    work.loc[vacio_obj, "_objetivo"] = "Sin objetivo (vacío)"
    work = work[work["_objetivo"] != "Sin objetivo (vacío)"].copy()

    # Score
    work["Porcentaje de consistencia"] = work.apply(
        lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1), axis=1
    )

    out = work.rename(columns={"_objetivo":"Objetivo específico","_actividad":"Actividad"})
    out["Fuente (archivo)"] = top_obj_label or ""
    return out[["Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)"]]

# ===========================
# Sugerencia de objetivo óptimo
# ===========================
def suggest_objective_for_activity(activity: str, candidates: List[str]) -> Tuple[str, float]:
    if is_blank(activity) or not candidates:
        return ("", 0.0)
    best_obj, best_score = "", -1.0
    for obj in candidates:
        s = combined_score(activity, obj) * 100.0
        if s > best_score:
            best_score = s
            best_obj = obj
    return (best_obj, round(float(best_score), 1))

# ===========================
# Informe Word avanzado (v9)
# ===========================
def add_table(doc: Document, headers: List[str], rows: List[List[str]]):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light List Accent 1"
    hdr = t.rows[0].cells
    for j, h in enumerate(headers):
        hdr[j].text = str(h)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = str(val)

def build_word_report(final: pd.DataFrame, n_archivos: int) -> bytes:
    doc = Document()
    doc.add_heading("Conclusiones de Consistencia de actividades – Informe avanzado", 0)

    # ------------------ Métricas globales ------------------
    n = len(final)
    mean = round(float(final["Porcentaje de consistencia"].mean()), 1) if n else 0.0
    median = round(float(final["Porcentaje de consistencia"].median()), 1) if n else 0.0
    p25 = round(float(final["Porcentaje de consistencia"].quantile(0.25)), 1) if n else 0.0
    p75 = round(float(final["Porcentaje de consistencia"].quantile(0.75)), 1) if n else 0.0
    vmin = round(float(final["Porcentaje de consistencia"].min()), 1) if n else 0.0
    vmax = round(float(final["Porcentaje de consistencia"].max()), 1) if n else 0.0

    doc.add_paragraph(f"Archivos procesados: {n_archivos}")
    doc.add_paragraph(f"Cantidad de actividades evaluadas: {n}")
    doc.add_paragraph(f"Porcentaje promedio de consistencia general: {mean:.1f}%")
    doc.add_paragraph(f"Mediana: {median:.1f}% | P25: {p25:.1f}% | P75: {p75:.1f}% | Mín/Máx: {vmin:.1f}% / {vmax:.1f}%")

    # Niveles
    alta = int((final["Porcentaje de consistencia"] >= 75).sum())
    media = int(((final["Porcentaje de consistencia"] >= 50) & (final["Porcentaje de consistencia"] < 75)).sum())
    baja  = int((final["Porcentaje de consistencia"] < 50).sum())

    doc.add_heading("Distribución por niveles", level=1)
    add_table(doc, ["Nivel","N actividades"], [
        ["Alta (>=75%)", alta],
        ["Media (50–74%)", media],
        ["Baja (<50%)", baja]
    ])
    doc.add_paragraph(
        "Interpretación: una mayor proporción en niveles Medio/Alto sugiere redacciones alineadas con los verbos, ámbitos y productos de los objetivos. "
        "Una concentración en Bajo indica redacciones genéricas u objetivos poco acotados."
    )

    # ------------------ Ranking por Objetivo específico ------------------
    doc.add_heading("Rendimiento por Objetivo específico (Top 10 críticos)", level=1)
    grp = final.groupby("Objetivo específico").agg(
        n=("Actividad","count"),
        mean=("Porcentaje de consistencia","mean"),
        median=("Porcentaje de consistencia","median"),
        bajo=("Porcentaje de consistencia", lambda s: (s < 50).mean()*100.0)
    ).reset_index()
    grp["mean"] = grp["mean"].round(1)
    grp["median"] = grp["median"].round(1)
    grp["bajo"] = grp["bajo"].round(1)
    worst = grp.sort_values(["mean","bajo","n"], ascending=[True, False, False]).head(10)
    rows = [[r["Objetivo específico"], int(r["n"]), f'{r["mean"]:.1f}%', f'{r["median"]:.1f}%', f'{r["bajo"]:.1f}%'] for _, r in worst.iterrows()]
    add_table(doc, ["Objetivo específico","N","Promedio","Mediana","% en Bajo"], rows)
    doc.add_paragraph(
        "Estos objetivos requieren priorización para revisar definiciones, ajustar verbos/resultados esperados y asegurar la trazabilidad con actividades."
    )

    # ------------------ Objetivos con mayor dispersión ------------------
    doc.add_heading("Objetivos con mayor dispersión interna", level=1)
    disp = final.groupby("Objetivo específico")["Porcentaje de consistencia"].agg(
        std=lambda s: float(np.std(s, ddof=0)),
        iqr=lambda s: float(s.quantile(0.75) - s.quantile(0.25)),
        n="count"
    ).reset_index()
    disp["std"] = disp["std"].round(1)
    disp["iqr"] = disp["iqr"].round(1)
    disp = disp.sort_values(["iqr","std"], ascending=False).head(8)
    rows = [[r["Objetivo específico"], int(r["n"]), f'{r["std"]:.1f}', f'{r["iqr"]:.1f}'] for _, r in disp.iterrows()]
    add_table(doc, ["Objetivo específico","N","Desvío estándar","IQR"], rows)
    doc.add_paragraph(
        "Alta dispersión sugiere criterios heterogéneos o actividades redactadas con niveles de especificidad muy dispares."
    )

    # ------------------ Actividades con alto potencial de mejora ------------------
    doc.add_heading("Actividades con alto potencial de mejora/reubicación", level=1)
    # Se consideran candidatas: actual <50% y mejora sugerida (delta) >= 15 p.p., si existen columnas de sugerencia
    cols_needed = {"Objetivo sugerido (máxima consistencia)","Porcentaje de consistencia (sugerido)","Diferencia (p.p.)"}
    if cols_needed.issubset(set(final.columns)):
        cand = final[(final["Porcentaje de consistencia"] < 50) & (final["Diferencia (p.p.)"] >= 15)].copy()
        cand = cand.sort_values(["Diferencia (p.p.)","Porcentaje de consistencia"], ascending=[False, True]).head(20)
        rows = [
            [
                r["Actividad"],
                r["Objetivo específico"],
                f'{float(r["Porcentaje de consistencia"]):.1f}%',
                r["Objetivo sugerido (máxima consistencia)"],
                f'{float(r["Porcentaje de consistencia (sugerido)"]):.1f}%',
                f'{float(r["Diferencia (p.p.)"]):.1f}'
            ]
            for _, r in cand.iterrows()
        ]
        add_table(doc, ["Actividad","Obj. actual","% actual","Obj. sugerido","% sugerido","Δ p.p."], rows)
        doc.add_paragraph(
            "Nota: la reubicación debe considerarse luego de **reelaborar la redacción** de la actividad. "
            "Si, tras la reescritura, la diferencia permanece alta y coherente con indicadores, recién ahí conviene moverla."
        )
    else:
        doc.add_paragraph("No se incluyeron columnas de sugerencia en el análisis actual; omitiendo esta sección.")

    # ------------------ Duplicadas/similares ------------------
    doc.add_heading("Actividades duplicadas o muy similares (indicio)", level=1)
    norm = final["Actividad"].astype(str).apply(normalize_for_dupes)
    dupemap = norm.value_counts()
    dups = dupemap[dupemap >= 2].head(10)
    if len(dups) == 0:
        doc.add_paragraph("No se detectaron duplicidades evidentes por normalización simple.")
    else:
        rows = [[k, int(v)] for k, v in dups.items()]
        add_table(doc, ["Actividad (normalizada)","Repeticiones"], rows)
        doc.add_paragraph(
            "Sugerencia: consolidar duplicadas como **líneas de trabajo** con sub-tareas medibles; "
            "evita dispersión y mejora la trazabilidad del PEI."
        )

    # ------------------ Recomendaciones y plan ------------------
    doc.add_heading("Guía práctica de reescritura", level=1)
    doc.add_paragraph("Plantilla sugerida: Verbo operativo + Objeto + Ámbito/Población + Entregable + Resultado esperado.")
    for ejemplo in [
        "Capacitar a 50 docentes de grado en evaluación por competencias (5 talleres, Q2) → Docentes formados y plan de aplicación.",
        "Implementar tablero de seguimiento en Looker Studio para objetivos 1.x (actualización mensual) → Indicadores disponibles y monitoreados.",
        "Diseñar e institucionalizar protocolo de autoevaluación anual (versión 1.0, Q3) → Informe de autoevaluación y plan de mejora."
    ]:
        doc.add_paragraph("• " + ejemplo)

    doc.add_heading("Plan de mejora por etapas", level=1)
    for item in [
        "Corto plazo (0–30 días): higiene de redacción, plantillas y glosario de verbos/entregables.",
        "Mediano plazo (1–3 meses): reencuadre de objetivos ambiguos, consolidación de duplicadas y trazabilidad KPI/evidencias.",
        "Revisión trimestral/semestral: correr calculadora, identificar ‘Baja’, reelaborar y volver a medir; gobernanza a través de un comité PEI."
    ]:
        doc.add_paragraph("• " + item)

    # ------------------ Anexo metodológico ------------------
    doc.add_heading("Anexo metodológico (síntesis)", level=1)
    doc.add_paragraph(
        "La consistencia se estima combinando similitud léxica (Token Set Ratio, 60%) y solapamiento de términos (Jaccard, 40%). "
        "Se limpia el ‘Objetivo específico’ para mantener solo el tramo ‘1.x …’, evitando mezclar con actividades o resultados."
    )

    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    return buf.getvalue()

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v9)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v9)")
st.caption("Acepta hasta 6 planillas (CSV/XLSX), limpia el objetivo (solo '1.x …'), excluye 'Sin objetivo (vacío)'; sugiere objetivo óptimo y genera **informe Word avanzado**.")

uploads = st.file_uploader(
    "Subí las planillas (p. ej.: 'Plan Estratégico ... Objetivo 1_Tabla.csv' ... 'Objetivo 6_Tabla.csv')",
    type=["csv","xlsx","xls"],
    accept_multiple_files=True
)

if uploads:
    st.write(f"Archivos cargados: **{len(uploads)}**")
    resultados = []
    detalle_archivos = []
    for f in uploads:
        label = parse_top_objective_from_name(getattr(f, "name", ""))
        frames = load_frames_from_upload(f)
        used = 0
        for df in frames:
            out = evaluate_df(df, label)
            if len(out):
                resultados.append(out)
                used += 1
        detalle_archivos.append((getattr(f, "name", "archivo"), label or "", used))

    if not resultados:
        st.warning("No se detectaron columnas de Objetivo/Actividad en los archivos cargados.")
    else:
        final = pd.concat(resultados, ignore_index=True)

        # -------- Sugerir objetivo óptimo por actividad --------
        candidatos = sorted(pd.Series(final["Objetivo específico"].unique()).dropna().tolist())
        sugeridos, sugeridos_pct, delta_pp = [], [], []
        for _, r in final.iterrows():
            best_obj, best_pct = suggest_objective_for_activity(r["Actividad"], candidatos)
            sugeridos.append(best_obj)
            sugeridos_pct.append(best_pct)
            delta_pp.append(round(best_pct - float(r["Porcentaje de consistencia"]), 1))
        final["Objetivo sugerido (máxima consistencia)"] = sugeridos
        final["Porcentaje de consistencia (sugerido)"] = sugeridos_pct
        final["Diferencia (p.p.)"] = delta_pp

        # -------- Métricas globales --------
        promedio = round(float(final["Porcentaje de consistencia"].mean()), 1) if len(final) else 0.0

        st.success(f"Se consolidaron **{len(final)}** actividades. Promedio global: **{promedio:.1f}%**.")
        st.dataframe(final.head(100))

        # -------- Excel: dos hojas --------
        informe = final[[
            "Objetivo específico","Actividad",
            "Porcentaje de consistencia",
            "Objetivo sugerido (máxima consistencia)",
            "Porcentaje de consistencia (sugerido)",
            "Diferencia (p.p.)"
        ]].copy()
        informe["Porcentaje de consistencia total promedio"] = promedio

        informe_fuente = final[[
            "Fuente (archivo)","Objetivo específico","Actividad",
            "Porcentaje de consistencia",
            "Objetivo sugerido (máxima consistencia)",
            "Porcentaje de consistencia (sugerido)",
            "Diferencia (p.p.)"
        ]].copy()

        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as w:
            informe.to_excel(w, index=False, sheet_name="Informe")
            informe_fuente.to_excel(w, index=False, sheet_name="Informe+Fuente")
        st.download_button("⬇️ Descargar Excel (consolidado)", data=buf_xlsx.getvalue(),
                           file_name="informe_consistencia_pei_consolidado.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # -------- Word avanzado --------
        word_bytes = build_word_report(final, n_archivos=len(uploads))
        st.download_button("⬇️ Descargar Word (Informe avanzado)", data=word_bytes,
                           file_name="conclusiones_consistencia_actividades_avanzado.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # -------- Tabla de archivos procesados --------
        st.subheader("Archivos procesados")
        st.table(pd.DataFrame(detalle_archivos, columns=["Archivo","Etiqueta detectada","Hojas utilizadas"]))
else:
    st.info("Cargá entre 1 y 6 archivos (CSV/XLSX). Si el nombre contiene 'Objetivo 1..6', se usa como etiqueta de fuente.")


