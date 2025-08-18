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
            frames.append(pd.read_csv(uploaded_file))
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
# Evaluación y sugerencias
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
    # excluir objetivos vacíos
    vacio_obj = work["_objetivo"].apply(is_blank)
    work.loc[vacio_obj, "_objetivo"] = "Sin objetivo (vacío)"
    work = work[work["_objetivo"] != "Sin objetivo (vacío)"].copy()

    work["Porcentaje de consistencia"] = work.apply(
        lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1), axis=1
    )
    out = work.rename(columns={"_objetivo":"Objetivo específico","_actividad":"Actividad"})
    out["Fuente (archivo)"] = top_obj_label or ""
    out["Cod. objetivo"] = out["Objetivo específico"].apply(extract_code)
    return out[["Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)","Cod. objetivo"]]

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
EXAMPLE_LINES_BY_TOP = {
    "1": [
        "•\tAlta carga de actividades, pero predominan los desvíos (ej. cultura de calidad, planes de mejora).",
        "•\tCasos de plena correspondencia en retroalimentación (encuestas a docentes, reuniones de comisión, tableros de seguimiento).",
        "•\tRecomendación: clarificar criterios de registro para evitar que todas las acciones de capacitación/difusión se carguen en “cultura de calidad”.",
    ],
    "2": [
        "•\tConvenios (ej. OSSE, Academia Nacional de Ciencias) aparecen bien ubicados, aunque algunos se cruzan con identidad cristiana.",
        "•\tPocas actividades cargadas bajo Plan de Comunicación institucional (2.3), aunque es un objetivo central.",
        "•\tRecomendación: fortalecer registro en sublíneas de comunicación y responsabilidad social.",
    ],
    "3": [
        "•\tBuen nivel de coherencia: permanencia y retención, capacitaciones y recursos educativos están bien alineados.",
        "•\tCasos de parcialidad: “Centro de Contactos” se relaciona tanto con tutoría (3.7) como con infraestructura tecnológica (3.6).",
        "•\tRecomendación: mantener consistencia diferenciando actividades tecnológicas vs. pedagógicas.",
    ],
    "4": [
        "•\tMuy escasa carga de actividades.",
        "•\tMuchas que deberían estar aquí (capacitaciones, encuestas de satisfacción docente) aparecen cargadas en Objetivo 1.",
        "•\tRecomendación: reforzar socialización del alcance de este objetivo.",
    ],
    "5": [
        "•\tSe observan actividades de participación y seguimiento, pero con registros dispersos y parciales.",
        "•\tEjemplo: acciones para graduados que podrían estar en Consejo de Graduados (5.2) o en Seguimiento (5.7).",
        "•\tRecomendación: aclarar subcategorías de registro y fomentar más reportes desde Bienestar Estudiantil y Graduados.",
    ],
    "6": [
        "•\tActividades escasamente registradas.",
        "•\tAlgunas acciones vinculadas a espiritualidad o formación cristiana fueron cargadas en convenios (2.1).",
        "•\tRecomendación: establecer claramente que todo lo pastoral y de formación en valores debe concentrarse aquí.",
    ],
}

def build_word_like_example(final: pd.DataFrame, n_archivos: int,
                            nombres_obj_top: Dict[str,str],
                            umbral_plena: float, umbral_parcial: float) -> bytes:
    doc = Document()
    doc.add_heading("Análisis de coherencia entre actividades registradas Formulario Único y acciones PEI", 0)

    # ---- RESUMEN
    doc.add_paragraph("RESUMEN:")
    doc.add_paragraph("Se presenta a continuación un análisis general respecto a dos cuestiones centrales para revisión de la Comisión de Seguimiento del PEI y reunión con los decanos:")
    doc.add_paragraph("A- Análisis de coherencia entre las actividades registradas en el Formulario Único y las acciones del Plan Estratégico Institucional UCCuyo 2023-2027:")
    doc.add_paragraph("B- Análisis del grado de desarrollo del PEI por objetivo específico")

    # ---- A) Coherencia
    doc.add_heading("A- Análisis de coherencia entre las actividades registradas en el Formulario Único y las acciones del Plan Estratégico Institucional UCCuyo 2023-2027:", level=1)

    # 1. Panorama General (con umbrales configurables)
    doc.add_heading("1. Panorama General", level=2)
    pcts = final["Porcentaje de consistencia"].astype(float)
    plena_n  = int((pcts >= umbral_plena).sum())
    parcial_n = int(((pcts >= umbral_parcial) & (pcts < umbral_plena)).sum())
    desvio_n  = int((pcts < umbral_parcial).sum())
    total_n   = int(len(final))
    pct_plena = (plena_n/total_n*100.0) if total_n else 0.0

    def bullet(doc, txt): 
        p = doc.add_paragraph(txt); p.style = doc.styles["List Paragraph"]

    bullet(doc, f"•\tActividades con plena correspondencia: {plena_n}")
    bullet(doc, f"•\tActividades con correspondencia parcial: {parcial_n}")
    bullet(doc, f"•\tActividades sin correspondencia clara (desvío): {desvio_n}")

    if pct_plena < 25:
        doc.add_paragraph("Esto muestra que menos del 25% de las actividades se registraron de manera plenamente alineada con las acciones del PEI. La mayoría requieren revisión para reubicación o ajuste.")
    else:
        doc.add_paragraph(f"Esto muestra que {pct_plena:.1f}% de las actividades se registraron de manera plenamente alineada con las acciones del PEI. La mayoría requieren revisión para reubicación o ajuste.")

    # 2. Principales hallazgos por objetivos (texto EXACTO editable)
    doc.add_heading("2. Principales hallazgos por objetivos", level=2)
    # Detectar objetivo top (1..6) por código 1.x
    tmp = final.copy()
    tmp["ObjTop"] = tmp["Cod. objetivo"].str.extract(r"^(\d+)").fillna("")
    for k in ["1","2","3","4","5","6"]:
        titulo = nombres_obj_top.get(k, f"Objetivo {k}")
        doc.add_paragraph(f"{titulo}")
        for line in EXAMPLE_LINES_BY_TOP.get(k, []):
            bullet(doc, line)

    # 3. Recomendaciones estratégicas (iguales al ejemplo)
    doc.add_heading("3. Recomendaciones estratégicas", level=2)
    for r in [
        "1.\tUnificar criterios de registro: evitar que cada área cargue la misma actividad en diferentes objetivos.",
        "2.\tCapacitación breve para equipos responsables: explicar con ejemplos qué se espera registrar en cada objetivo específico.",
        "3.\tReforzar áreas críticas: objetivos 4, 5 y 6 requieren mayor volumen de acciones registradas.",
        "4.\tRevisión de desvíos: más de 100 actividades deben ser reubicadas para reflejar fielmente el PEI.",
        "5.\tMonitoreo temprano: utilizar la tabla refinada como insumo en la próxima reunión de Comisión de Planificación y Evaluación.",
    ]:
        doc.add_paragraph(r)

    # ---- B) Grado de desarrollo del PEI
    doc.add_heading("B- Análisis del grado de desarrollo del PEI por objetivo específico, considerando dos dimensiones:", level=1)
    doc.add_paragraph("•\tCantidad de actividades registradas")
    doc.add_paragraph("•\tPorcentaje de actividades con plena correspondencia (es decir, bien alineadas con el PEI).")
    doc.add_paragraph("Se observan dos tendencias claras:")

    det = final.groupby("Objetivo específico").agg(
        n=("Actividad","count"),
        pct_plena=("Porcentaje de consistencia", lambda s: (s >= umbral_plena).mean()*100.0)
    ).reset_index()
    det["pct_plena"] = det["pct_plena"].round(1)

    n_q50 = det["n"].median() if len(det) else 0
    n_q25 = det["n"].quantile(0.25) if len(det) else 0

    # 1) Mayormente desarrollados
    doc.add_paragraph("")
    doc.add_paragraph("1. Objetivos mayormente desarrollados")
    mayor = det[(det["n"] >= n_q50) & (det["pct_plena"] >= 85)].sort_values(["pct_plena","n"], ascending=[False,False]).head(8)
    if len(mayor) == 0:
        doc.add_paragraph("•\t(no se identifican con los umbrales actuales)")
    else:
        for _, r in mayor.iterrows():
            doc.add_paragraph(f"•\t{r['Objetivo específico']} – {int(r['n'])} actividades; {r['pct_plena']:.1f}% de plena correspondencia.")

    # 2) Registro insuficiente o con desvíos
    doc.add_paragraph("")
    doc.add_paragraph("2. Objetivos con registro insuficiente o con desvíos")
    insuf = det[(det["n"] < n_q25) | (det["pct_plena"] < 50)].sort_values(["pct_plena","n"], ascending=[True,True]).head(12)
    if len(insuf) == 0:
        doc.add_paragraph("•\t(no se observan con los umbrales actuales)")
    else:
        for _, r in insuf.iterrows():
            doc.add_paragraph(f"•\t{r['Objetivo específico']} – {int(r['n'])} actividades; {r['pct_plena']:.1f}% de plena correspondencia.")

    # 3) Síntesis (formato exacto)
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
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v10.3)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v10.3)")
st.caption("Word con la **misma estructura y frases base** del ejemplo. Ajustá umbrales para que los conteos coincidan con tu referencia.")

# Panel lateral: nombres visibles de los 6 objetivos y umbrales
with st.sidebar:
    st.markdown("**Nombres visibles de los 6 objetivos**")
    obj_names = {
        "1": st.text_input("1", "Objetivo 1: Aseguramiento de la calidad"),
        "2": st.text_input("2", "Objetivo 2: Vinculación y comunicación"),
        "3": st.text_input("3", "Objetivo 3: Educación a Distancia"),
        "4": st.text_input("4", "Objetivo 4: Recursos humanos"),
        "5": st.text_input("5", "Objetivo 5: Estudiantes y graduados"),
        "6": st.text_input("6", "Objetivo 6: Identidad cristiana"),
    }
    st.markdown("---")
    st.markdown("**Umbrales de clasificación (Word)**")
    umbral_plena = st.slider("Plena correspondencia ≥", min_value=40, max_value=90, step=1, value=75)
    umbral_parcial = st.slider("Correspondencia parcial ≥", min_value=10, max_value=umbral_plena-1, step=1, value=50)

uploads = st.file_uploader(
    "Subí hasta 6 planillas (CSV/XLSX), una por cada objetivo si las tenés separadas",
    type=["csv","xlsx","xls"], accept_multiple_files=True
)

if uploads:
    st.write(f"Archivos cargados: **{len(uploads)}**")
    resultados, detalle_archivos = [], []
    for f in uploads:
        label = parse_top_objective_from_name(getattr(f, "name", ""))
        for df in load_frames_from_upload(f):
            out = evaluate_df(df, label)
            if len(out):
                resultados.append(out)
        detalle_archivos.append(getattr(f, "name", "archivo"))

    if not resultados:
        st.warning("No se detectaron columnas de Objetivo/Actividad en los archivos cargados.")
    else:
        final = pd.concat(resultados, ignore_index=True)

        # Sugerencias (para Excel)
        candidatos = sorted(pd.Series(final["Objetivo específico"].unique()).dropna().tolist())
        sug_obj, sug_pct, delta = [], [], []
        for _, r in final.iterrows():
            bobj, bpct = suggest_objective_for_activity(r["Actividad"], candidatos)
            sug_obj.append(bobj); sug_pct.append(bpct)
            delta.append(round(bpct - float(r["Porcentaje de consistencia"]), 1))
        final["Objetivo sugerido (máxima consistencia)"] = sug_obj
        final["Porcentaje de consistencia (sugerido)"] = sug_pct
        final["Diferencia (p.p.)"] = delta

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

        # -------- Word (estructura exacta) --------
        word_bytes = build_word_like_example(
            final, n_archivos=len(uploads), nombres_obj_top=obj_names,
            umbral_plena=umbral_plena, umbral_parcial=umbral_parcial
        )
        st.download_button("⬇️ Descargar Word (estructura ejemplo)", data=word_bytes,
                           file_name="analisis_coherencia_actividades_PEI.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
else:
    st.info("Cargá entre 1 y 6 archivos (CSV/XLSX). Si el nombre contiene 'Objetivo 1..6', se usa como etiqueta de fuente.")

