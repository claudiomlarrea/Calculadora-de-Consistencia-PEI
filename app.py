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
    # Objetivo específico
    col_obj = best_column(df, [
        "objetivo especifico", "objetivo específico", "objetivos especificos", "objetivos específicos",
        "objetivo", "objetivo pei", "objetivo del pei", "objetivos especificos 1", "objetivos especificos 2",
        "objetivos específicos 1", "objetivos específicos 2"
    ])
    # Actividad
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
        # Fallback a Excel simple
        frames.append(pd.read_excel(uploaded_file))
    return frames

# ===========================
# Evaluación por DataFrame
# ===========================
def evaluate_df(df: pd.DataFrame, top_obj_label: Optional[str], combine_code_and_text: bool=False) -> pd.DataFrame:
    col_obj, col_act = guess_columns(df)
    if not col_obj or not col_act:
        return pd.DataFrame(columns=["Objetivo específico","Actividad","consistencia_%","Fuente (archivo)"])

    objetivo = to_clean_str_series(df[col_obj]).apply(force_goal_only)
    actividad = to_clean_str_series(df[col_act])

    work = pd.DataFrame({
        "_objetivo": objetivo,
        "_actividad": actividad
    })

    # Limpiar y excluir objetivos vacíos
    vacio_obj = work["_objetivo"].apply(is_blank)
    work.loc[vacio_obj, "_objetivo"] = "Sin objetivo (vacío)"
    work = work[work["_objetivo"] != "Sin objetivo (vacío)"].copy()

    # Score
    work["consistencia_%"] = work.apply(lambda r: round(combined_score(r["_actividad"], r["_objetivo"]) * 100.0, 1), axis=1)

    out = work.rename(columns={"_objetivo":"Objetivo específico","_actividad":"Actividad"})
    if top_obj_label:
        out["Fuente (archivo)"] = top_obj_label
    else:
        out["Fuente (archivo)"] = ""
    return out[["Objetivo específico","Actividad","consistencia_%","Fuente (archivo)"]]

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

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Análisis de consistencia – Multi-archivo (v8)", layout="wide")
st.title("📊 Análisis de consistencia de actividades PEI – Multi-archivo (v8)")
st.caption("Subí hasta 6 archivos (CSV/XLSX), uno por cada objetivo. El sistema consolida, limpia el objetivo (solo '1.x …') y calcula la consistencia por actividad. Excluye 'Sin objetivo (vacío)'.")

uploads = st.file_uploader(
    "Sube las planillas (por ejemplo: 'Plan Estratégico ... Objetivo 1_Tabla.csv' ... 'Objetivo 6_Tabla.csv')",
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
        st.warning("No se pudieron detectar columnas de Objetivo/Actividad en los archivos cargados.")
    else:
        final = pd.concat(resultados, ignore_index=True)

        # Métricas
        promedio = round(float(final["consistencia_%"].mean()), 1)
        alta = int((final["consistencia_%"] >= 75).sum())
        media = int(((final["consistencia_%"] >= 50) & (final["consistencia_%"] < 75)).sum())
        baja  = int((final["consistencia_%"] < 50).sum())

        st.success(f"Se consolidaron **{len(final)}** actividades. Promedio global: **{promedio:.1f}%**.")
        st.dataframe(final.head(100))

        # -------- Excel: dos hojas --------
        # Hoja 1: Informe (4 columnas)
        informe = final[["Objetivo específico","Actividad","consistencia_%"]].copy()
        informe["Promedio global"] = promedio
        informe = informe.rename(columns={
            "consistencia_%": "Porcentaje de correlación o consistencia de cada actividad",
            "Promedio global": "Porcentaje de correlación total promedio"
        })
        # Hoja 2: Informe + Fuente (para rastreo)
        informe_fuente = final[["Fuente (archivo)","Objetivo específico","Actividad","consistencia_%"]].copy()
        informe_fuente = informe_fuente.rename(columns={"consistencia_%":"Porcentaje (actividad) %"})

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







