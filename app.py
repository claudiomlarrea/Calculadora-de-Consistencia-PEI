import io
import re
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
        "objetivo", "objetivo pei", "objetivo del pei", "objetivos especificos 1", "objetivos especificos 2",
        "objetivos específicos 1", "objetivos específicos 2"
    ])
    col_act = best_column(df, [
        "actividad", "actividades", "acciones", "actividades objetivo", "actividades objetivo 1",
        "actividad especifica", "actividad específica", "descripcion de la actividad", "descripción de la actividad"
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
        return pd.DataFrame(columns=["Objetivo específico","Actividad","Porcentaje de consistencia","Fuente (archivo)"])

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
    """
    Devuelve (objetivo_sugerido, score_sugerido) que maximiza la consistencia para 'activity'
    entre la lista de 'candidates'.
    """
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
# Informe Word
# ===========================
def generar_informe_word(n_acts: int, promedio: float, dist: Dict[str,int], n_archivos: int) -> bytes:
    doc = Document()
    doc.add_heading("Conclusiones de Consistencia de actividades", 0)

    p = doc.add_paragraph()
    p.add_run("Cantidad de actividades evaluadas: ").bold = True
    p.add_run(str(n_acts))

    p = doc.add_paragraph()
    p.add_run("Porcentaje promedio de consistencia general: ").bold = True
    p.add_run(f"{promedio:.1f}%")

    p = doc.add_paragraph()
    p.add_run("Archivos procesados: ").bold = True
    p.add_run(str(n_archivos))

    doc.add_heading("Interpretación de los resultados", level=1)
    if promedio >= 75:
        doc.add_paragraph(
            "El promedio global indica una consistencia alta entre actividades y objetivos específicos. "
            "Las descripciones de actividades reflejan con claridad el aporte al PEI. "
            "Se recomienda documentar y estandarizar las buenas prácticas detectadas."
        )
    elif promedio >= 50:
        doc.add_paragraph(
            "La consistencia es intermedia. Hay áreas fuertes y otras con desajustes (actividades genéricas o productos poco definidos). "
            "Conviene revisar objetivos con menor consistencia y reubicar actividades según las sugerencias de esta calculadora."
        )
    else:
        doc.add_paragraph(
            "La consistencia global es baja; se observan actividades que no reflejan suficientemente su aporte a los objetivos del PEI. "
            "Se sugiere reescribir actividades y reubicarlas en el objetivo con mayor correlación."
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
        "Incluir entregable/resultado y el ámbito/población objetivo.",
        "Evitar duplicados y actividades demasiado genéricas; convertirlas en líneas de trabajo con sub-tareas medibles.",
        "Reubicar actividades según el **Objetivo sugerido** cuando el delta de consistencia sea relevante."
    ]:
        doc.add_paragraph("• " + r)

    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    return buf.getvalue()

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v8.1)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v8.1)")
st.caption("Acepta hasta 6 planillas (CSV/XLSX), limpia el objetivo (solo '1.x …'), excluye 'Sin objetivo (vacío)' y sugiere el objetivo con **mayor consistencia** para cada actividad.")

uploads = st.file_uploader(
    "Subí las planillas (por ejemplo: 'Plan Estratégico ... Objetivo 1_Tabla.csv' ... 'Objetivo 6_Tabla.csv')",
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

        # Métricas
        promedio = round(float(final["Porcentaje de consistencia"].mean()), 1)
        alta = int((final["Porcentaje de consistencia"] >= 75).sum())
        media = int(((final["Porcentaje de consistencia"] >= 50) & (final["Porcentaje de consistencia"] < 75)).sum())
        baja  = int((final["Porcentaje de consistencia"] < 50).sum())

        st.success(f"Se consolidaron **{len(final)}** actividades. Promedio global: **{promedio:.1f}%**.")
        st.dataframe(final.head(100))

        # -------- Excel: dos hojas --------
        # Hoja 1: Informe (incluye objetivo sugerido)
        informe = final[[
            "Objetivo específico","Actividad",
            "Porcentaje de consistencia",
            "Objetivo sugerido (máxima consistencia)",
            "Porcentaje de consistencia (sugerido)",
            "Diferencia (p.p.)"
        ]].copy()
        informe["Porcentaje de consistencia total promedio"] = promedio

        # Hoja 2: Informe + Fuente (trazabilidad)
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

        # -------- Word: conclusiones --------
        dist = {"Alta (>=75%)": alta, "Media (50–74%)": media, "Baja (<50%)": baja}
        word_bytes = generar_informe_word(n_acts=len(final), promedio=promedio, dist=dist, n_archivos=len(uploads))
        st.download_button("⬇️ Descargar Word (Conclusiones)", data=word_bytes,
                           file_name="conclusiones_consistencia_actividades.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # -------- Tabla de archivos procesados --------
        st.subheader("Archivos procesados")
        st.table(pd.DataFrame(detalle_archivos, columns=["Archivo","Etiqueta detectada","Hojas utilizadas"]))
else:
    st.info("Cargá entre 1 y 6 archivos (CSV/XLSX). Si el nombre contiene 'Objetivo 1..6', se usa como etiqueta de fuente.")




