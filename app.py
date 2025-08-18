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

def extract_code(obj_text: str) -> str:
    m = CODE_RE.search(str(obj_text))
    return m.group(0) if m else ""

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

    work = pd.DataFrame({"_objetivo": objetivo, "_actividad": actividad})

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
# Informe Word – estructura EXACTA del ejemplo
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

def text_bullet(doc: Document, txt: str):
    p = doc.add_paragraph(txt)
    p.style = doc.styles["List Paragraph"]

def build_word_like_example(final: pd.DataFrame, n_archivos: int, nombres_obj_top: Dict[str,str]) -> bytes:
    """
    Genera un .docx con la misma estructura y redacción del ejemplo:
    - RESUMEN
    - A- Análisis de coherencia ...
      1. Panorama General (3 bullets con counts)
      2. Principales hallazgos por objetivos (1..6)
      3. Recomendaciones estratégicas (5 bullets)
    - B- Análisis del grado de desarrollo ...
      1) Objetivos mayormente desarrollados
      2) Objetivos con registro insuficiente o con desvíos
      3) Síntesis
    """
    doc = Document()
    doc.add_heading("Análisis de coherencia entre actividades registradas Formulario Único y acciones PEI", 0)

    # ---- RESUMEN
    doc.add_paragraph("RESUMEN:")
    doc.add_paragraph("Se presenta a continuación un análisis general respecto a dos cuestiones centrales para revisión de la Comisión de Seguimiento del PEI y reunión con los decanos:")
    doc.add_paragraph("A- Análisis de coherencia entre las actividades registradas en el Formulario Único y las acciones del Plan Estratégico Institucional UCCuyo 2023-2027:")
    doc.add_paragraph("B- Análisis del grado de desarrollo del PEI por objetivo específico")

    # ---- A) Coherencia
    doc.add_heading("A- Análisis de coherencia entre las actividades registradas en el Formulario Único y las acciones del Plan Estratégico Institucional UCCuyo 2023-2027:", level=1)

    # 1. Panorama General
    doc.add_heading("1. Panorama General", level=2)
    plena_n = int((final["Porcentaje de consistencia"] >= 75).sum())
    parcial_n = int(((final["Porcentaje de consistencia"] >= 50) & (final["Porcentaje de consistencia"] < 75)).sum())
    desvio_n = int((final["Porcentaje de consistencia"] < 50).sum())
    total_n = int(len(final))
    pct_plena = (plena_n / total_n * 100.0) if total_n else 0.0

    text_bullet(doc, f"•\tActividades con plena correspondencia: {plena_n}")
    text_bullet(doc, f"•\tActividades con correspondencia parcial: {parcial_n}")
    text_bullet(doc, f"•\tActividades sin correspondencia clara (desvío): {desvio_n}")
    doc.add_paragraph(
        f"Esto muestra que {pct_plena:.1f}% de las actividades se registraron de manera plenamente alineada con las acciones del PEI. "
        "La mayoría requieren revisión para reubicación o ajuste."
    )

    # 2. Principales hallazgos por objetivos (1..6)
    doc.add_heading("2. Principales hallazgos por objetivos", level=2)

    # Resumen por objetivo específico (código 1.x -> 1, etc.)
    tmp = final.copy()
    tmp["ObjTop"] = tmp["Cod. objetivo"].str.extract(r"^(\d+)").fillna("")
    grp = tmp.groupby("ObjTop").agg(
        n=("Actividad","count"),
        mean=("Porcentaje de consistencia","mean"),
        plena=("Porcentaje de consistencia", lambda s: (s >= 75).mean()*100.0),
        desvio=("Porcentaje de consistencia", lambda s: (s < 50).mean()*100.0)
    ).reset_index()
    grp["mean"] = grp["mean"].round(1)
    grp["plena"] = grp["plena"].round(1)
    grp["desvio"] = grp["desvio"].round(1)

    median_n = grp["n"].median() if len(grp) else 0

    def hallmark_lines(row) -> List[str]:
        lines = []
        name = nombres_obj_top.get(str(row["ObjTop"]), f"Objetivo {row['ObjTop']}")
        lines.append(f"{name}")
        if row["n"] >= median_n:
            lines.append("•\tAlta carga de actividades.")
        if row["desvio"] >= 50:
            lines.append("•\tPredominan los desvíos; revisar criterios de registro y redacción.")
        if row["plena"] >= 75:
            lines.append("•\tCasos de plena correspondencia con sublíneas específicas.")
        # Recomendación genérica por objetivo
        lines.append("•\tRecomendación: clarificar sublíneas y criterios para evitar mezcla de categorías.")
        return lines

    for _, r in grp.sort_values("ObjTop").iterrows():
        for line in hallmark_lines(r):
            doc.add_paragraph(line)

    # 3. Recomendaciones estratégicas
    doc.add_heading("3. Recomendaciones estratégicas", level=2)
    for r in [
        "1.\tUnificar criterios de registro: evitar que cada área cargue la misma actividad en diferentes objetivos.",
        "2.\tCapacitación breve para equipos responsables: explicar con ejemplos qué se espera registrar en cada objetivo específico.",
        "3.\tReforzar áreas críticas: objetivos con baja carga o alto desvío requieren mayor volumen y calidad de registro.",
        "4.\tRevisión de desvíos: reubicar actividades luego de su reelaboración textual para reflejar fielmente el PEI.",
        "5.\tMonitoreo temprano: usar la tabla consolidada como insumo en la próxima reunión de la Comisión de Planificación y Evaluación."
    ]:
        doc.add_paragraph(r)

    # ---- B) Grado de desarrollo
    doc.add_heading("B- Análisis del grado de desarrollo del PEI por objetivo específico, considerando dos dimensiones:", level=1)
    doc.add_paragraph("•\tCantidad de actividades registradas")
    doc.add_paragraph("•\tPorcentaje de actividades con plena correspondencia (es decir, bien alineadas con el PEI).")
    doc.add_paragraph("Se observan dos tendencias claras:")

    # Métricas por objetivo específico (sub-objetivo 1.x, 2.x, etc.)
    det = final.groupby("Objetivo específico").agg(
        n=("Actividad","count"),
        pct_plena=("Porcentaje de consistencia", lambda s: (s >= 75).mean()*100.0)
    ).reset_index()
    det["pct_plena"] = det["pct_plena"].round(1)

    # Umbrales
    n_q50 = det["n"].median() if len(det) else 0
    n_q25 = det["n"].quantile(0.25) if len(det) else 0

    # 1) Mayormente desarrollados: alta N y alta % plena
    doc.add_paragraph("")
    doc.add_paragraph("1. Objetivos mayormente desarrollados")
    mayor = det[(det["n"] >= n_q50) & (det["pct_plena"] >= 85)].sort_values(["pct_plena","n"], ascending=[False,False]).head(8)
    if len(mayor) == 0:
        doc.add_paragraph("•\t(no se identifican con los umbrales actuales)")
    else:
        for _, r in mayor.iterrows():
            doc.add_paragraph(f"•\t{r['Objetivo específico']} – {int(r['n'])} actividades; {r['pct_plena']:.1f}% de plena correspondencia.")

    # 2) Con registro insuficiente o con desvíos
    doc.add_paragraph("")
    doc.add_paragraph("2. Objetivos con registro insuficiente o con desvíos")
    insuf = det[(det["n"] < n_q25) | (det["pct_plena"] < 50)].sort_values(["pct_plena","n"], ascending=[True,True]).head(12)
    if len(insuf) == 0:
        doc.add_paragraph("•\t(no se observan con los umbrales actuales)")
    else:
        for _, r in insuf.iterrows():
            doc.add_paragraph(f"•\t{r['Objetivo específico']} – {int(r['n'])} actividades; {r['pct_plena']:.1f}% de plena correspondencia.")

    # 3) Síntesis
    doc.add_paragraph("")
    doc.add_paragraph("3. Síntesis")
    bien = mayor["Objetivo específico"].tolist()
    parcial = det[(det["pct_plena"].between(50, 74.9))]["Objetivo específico"].tolist()
    prioridad = insuf["Objetivo específico"].tolist()

    doc.add_paragraph("•\tBien desarrollados: " + (", ".join(bien) if bien else "—"))
    doc.add_paragraph("•\tCon desarrollo parcial pero disperso: " + (", ".join(parcial) if len(parcial) else "—"))
    doc.add_paragraph("•\tRequieren atención prioritaria: " + (", ".join(prioridad) if len(prioridad) else "—"))

    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    return buf.getvalue()

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v10.2)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v10.2)")
st.caption("Genera Word con la **misma estructura** del ejemplo (RESUMEN, A/B, hallazgos y síntesis). También entrega el Excel consolidado.")

# Nombres de objetivos top (1..6) editables para que el Word se lea igual que tu ejemplo
with st.sidebar:
    st.markdown("**Nombres de objetivos (1..6)**")
    obj_names = {
        "1": st.text_input("1", "Objetivo 1: Aseguramiento de la calidad"),
        "2": st.text_input("2", "Objetivo 2: Vinculación y comunicación"),
        "3": st.text_input("3", "Objetivo 3: Educación a Distancia"),
        "4": st.text_input("4", "Objetivo 4: Recursos humanos"),
        "5": st.text_input("5", "Objetivo 5: Estudiantes y graduados"),
        "6": st.text_input("6", "Objetivo 6: Identidad cristiana"),
    }

uploads = st.file_uploader(
    "Subí hasta 6 planillas (CSV/XLSX), una por cada objetivo si las tenés separadas",
    type=["csv","xlsx","xls"], accept_multiple_files=True
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
                resultados.append(out); used += 1
        detalle_archivos.append((getattr(f, "name", "archivo"), label or "", used))

    if not resultados:
        st.warning("No se detectaron columnas de Objetivo/Actividad en los archivos cargados.")
    else:
        final = pd.concat(resultados, ignore_index=True)

        # Sugerencia (se mantiene en Excel)
        candidatos = sorted(pd.Series(final["Objetivo específico"].unique()).dropna().tolist())
        sugeridos, sugeridos_pct, delta_pp = [], [], []
        for _, r in final.iterrows():
            best_obj, best_pct = suggest_objective_for_activity(r["Actividad"], candidatos)
            sugeridos.append(best_obj); sugeridos_pct.append(best_pct)
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

        # -------- Word con estructura del ejemplo --------
        word_bytes = build_word_like_example(final, n_archivos=len(uploads), nombres_obj_top=obj_names)
        st.download_button("⬇️ Descargar Word (estructura ejemplo)", data=word_bytes,
                           file_name="analisis_coherencia_actividades_PEI.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # Tabla de archivos procesados
        st.subheader("Archivos procesados")
        st.table(pd.DataFrame(detalle_archivos, columns=["Archivo","Etiqueta detectada","Hojas utilizadas"]))
else:
    st.info("Cargá entre 1 y 6 archivos (CSV/XLSX). Si el nombre contiene 'Objetivo 1..6', se usa como etiqueta de fuente.")

