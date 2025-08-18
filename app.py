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

def normalize_for_dupes(s: str) -> str:
    s = strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_code(obj_text: str) -> str:
    m = CODE_RE.search(str(obj_text))
    return m.group(0) if m else ""

# Clasificación por umbrales (alineado al estilo del ejemplo)
def corr_label(p: float) -> str:
    if p >= 75: return "plena"
    if p >= 50: return "parcial"
    return "desvío"  # sin correspondencia clara

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
    frames = []
    name = getattr(uploaded_file, "name", "archivo")
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            frames.append(df)
        else:
            xls = pd.ExcelFile(uploaded_file)
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
            "Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)","Cod. objetivo"
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
    out["Cod. objetivo"] = out["Objetivo específico"].apply(extract_code)
    return out[["Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)","Cod. objetivo"]]

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
# Informe Word – estilo “plantilla PEI”
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

def build_word_report_template(final: pd.DataFrame, n_archivos: int) -> bytes:
    doc = Document()
    doc.add_heading("Análisis de coherencia entre actividades registradas (Formulario Único) y acciones del PEI", 0)

    # ----- RESUMEN -----
    doc.add_paragraph("RESUMEN:")
    doc.add_paragraph(
        "Se presenta un análisis general respecto a dos cuestiones centrales para revisión de la Comisión de Seguimiento del PEI:"
    )
    doc.add_paragraph("A- Análisis de coherencia entre las actividades registradas en el Formulario Único y las acciones del PEI.")
    doc.add_paragraph("B- Análisis del grado de desarrollo del PEI por objetivo específico.")
    doc.add_paragraph(f"Archivos procesados: {n_archivos}")

    # Clasificación
    n = len(final)
    plena = int((final["Porcentaje de consistencia"] >= 75).sum())
    parcial = int(((final["Porcentaje de consistencia"] >= 50) & (final["Porcentaje de consistencia"] < 75)).sum())
    desvio = int((final["Porcentaje de consistencia"] < 50).sum())
    mean = round(float(final["Porcentaje de consistencia"].mean()), 1) if n else 0.0

    doc.add_paragraph("")
    doc.add_heading("A- Análisis de coherencia entre actividades y acciones del PEI", level=1)

    # 1) Panorama general
    doc.add_heading("1. Panorama General", level=2)
    add_table(doc, ["Categoría","N"], [
        ["Actividades con plena correspondencia", plena],
        ["Actividades con correspondencia parcial", parcial],
        ["Actividades sin correspondencia clara (desvío)", desvio],
    ])
    doc.add_paragraph(
        f"El promedio general de consistencia es {mean:.1f}%. "
        "La proporción de actividades en ‘plena correspondencia’ refleja el grado de alineación efectiva del registro con las acciones del PEI."
    )

    # 2) Principales hallazgos por objetivos (heurística)
    doc.add_heading("2. Principales hallazgos por objetivos", level=2)
    grp = final.groupby(["Objetivo específico","Cod. objetivo"]).agg(
        n=("Actividad","count"),
        mean=("Porcentaje de consistencia","mean"),
        plena=("Porcentaje de consistencia", lambda s: (s >= 75).mean()*100.0),
        desvio=("Porcentaje de consistencia", lambda s: (s < 50).mean()*100.0)
    ).reset_index()
    grp["mean"] = grp["mean"].round(1)
    grp["plena"] = grp["plena"].round(1)
    grp["desvio"] = grp["desvio"].round(1)

    # Ordenar por riesgo (más desvío, menor media, más casos)
    hall = grp.sort_values(["desvio","mean","n"], ascending=[False, True, False]).head(10)
    rows = [[r["Cod. objetivo"] or r["Objetivo específico"], int(r["n"]),
             f'{r["mean"]:.1f}%', f'{r["plena"]:.1f}%', f'{r["desvio"]:.1f}%']
            for _, r in hall.iterrows()]
    add_table(doc, ["Objetivo","N","Promedio","% Plena","% Desvío"], rows)

    # Recomendaciones estratégicas (alineadas a la plantilla)
    doc.add_heading("3. Recomendaciones estratégicas", level=2)
    for r in [
        "Unificar criterios de registro entre áreas para evitar duplicidades y desvíos.",
        "Capacitar brevemente a equipos responsables con ejemplos por objetivo específico.",
        "Reforzar objetivos subrepresentados (baja N) o con alto porcentaje de desvío.",
        "Revisar y reubicar actividades mal ubicadas tras su reelaboración textual."
    ]:
        doc.add_paragraph("• " + r)

    # ----- B -----
    doc.add_heading("B- Grado de desarrollo del PEI por objetivo específico", level=1)
    doc.add_paragraph("Se consideran dos dimensiones: (i) cantidad de actividades registradas y (ii) porcentaje de ‘plena correspondencia’.")

    # Métrica por objetivo
    dev = grp.copy()
    dev["pct_plena"] = dev["plena"]  # alias legible
    # Umbrales para categorizar
    q_n = dev["n"].quantile([0.25, 0.5, 0.75]).to_dict() if len(dev) else {0.25:0,0.5:0,0.75:0}

    # 1) Mayormente desarrollados: alta N y alta % plena
    mayormente = dev[(dev["n"] >= q_n.get(0.5,0)) & (dev["pct_plena"] >= 75)].sort_values(["pct_plena","n"], ascending=[False,False]).head(10)
    if len(mayormente):
        doc.add_heading("1. Objetivos mayormente desarrollados", level=2)
        rows = [[r["Cod. objetivo"] or r["Objetivo específico"], int(r["n"]), f'{r["pct_plena"]:.1f}%'] for _, r in mayormente.iterrows()]
        add_table(doc, ["Objetivo","N","% Plena"], rows)
    else:
        doc.add_paragraph("No se identifican objetivos claramente mayormente desarrollados según los umbrales actuales.")

    # 2) Con registro insuficiente o con desvíos: baja N o bajo % plena
    insuf = dev[(dev["n"] < q_n.get(0.25,0)) | (dev["pct_plena"] < 50)].sort_values(["pct_plena","n"], ascending=[True, True]).head(15)
    if len(insuf):
        doc.add_heading("2. Objetivos con registro insuficiente o con desvíos", level=2)
        rows = [[r["Cod. objetivo"] or r["Objetivo específico"], int(r["n"]), f'{r["pct_plena"]:.1f}%'] for _, r in insuf.iterrows()]
        add_table(doc, ["Objetivo","N","% Plena"], rows)

    # 3) Síntesis
    doc.add_heading("3. Síntesis", level=2)
    bien = dev[(dev["pct_plena"] >= 85) & (dev["n"] >= q_n.get(0.5,0))]["Cod. objetivo"].fillna(dev["Objetivo específico"]).tolist()
    parcial = dev[(dev["pct_plena"].between(50, 74.9))]["Cod. objetivo"].fillna(dev["Objetivo específico"]).tolist()
    prioridad = dev[(dev["pct_plena"] < 50) | (dev["n"] < q_n.get(0.25,0))]["Cod. objetivo"].fillna(dev["Objetivo específico"]).tolist()

    doc.add_paragraph("Bien desarrollados: " + (", ".join(bien) if bien else "—"))
    doc.add_paragraph("Con desarrollo parcial pero disperso: " + (", ".join(parcial) if parcial else "—"))
    doc.add_paragraph("Requieren atención prioritaria: " + (", ".join(prioridad) if prioridad else "—"))

    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    return buf.getvalue()

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v10)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v10)")
st.caption("Acepta hasta 6 planillas (CSV/XLSX), limpia el objetivo (solo '1.x …'), excluye 'Sin objetivo (vacío)'; Excel consolidado y **Word con el formato de la plantilla PEI**.")

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

        # Sugerencia de objetivo óptimo (se mantiene para Excel)
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

        # Métrica global
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

        # -------- Word – formato plantilla PEI --------
        word_bytes = build_word_report_template(final, n_archivos=len(uploads))
        st.download_button("⬇️ Descargar Word (formato plantilla PEI)", data=word_bytes,
                           file_name="analisis_coherencia_actividades_PEI.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # Tabla de archivos procesados
        st.subheader("Archivos procesados")
        st.table(pd.DataFrame(detalle_archivos, columns=["Archivo","Etiqueta detectada","Hojas utilizadas"]))
else:
    st.info("Cargá entre 1 y 6 archivos (CSV/XLSX). Si el nombre contiene 'Objetivo 1..6', se usa como etiqueta de fuente.")

