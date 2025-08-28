
import pandas as pd
import numpy as np
import io
import re
from typing import Dict, List

def _read_csv_flexible(file_obj):
    """Lee CSV desde un UploadedFile de Streamlit o file-like, con fallback de encoding."""
    raw = file_obj.read()
    for enc in (None, "utf-8", "latin1", "cp1252"):
        try:
            bio = io.BytesIO(raw)
            if enc:
                df = pd.read_csv(bio, encoding=enc)
            else:
                df = pd.read_csv(bio)
            return df
        except UnicodeDecodeError:
            continue
    bio = io.BytesIO(raw)
    return pd.read_csv(bio, encoding="latin1", errors="ignore")

def _clean_val(x):
    if isinstance(x, str):
        x = x.strip()
        if x == "-" or x.lower() in {"", "na", "n/a", "null"}:
            return np.nan
    return x

def _detect_objetivo_from_columns(columns: List[str]) -> int:
    cols = [c.strip() for c in columns]
    for obj_n in range(1, 7):
        if f"Objetivos específicos {obj_n}" in cols and f"Actividades Objetivo {obj_n}" in cols:
            return obj_n
    # aceptar variantes con espacios o numeración en el nombre
    for obj_n in range(1, 7):
        maybe = [c for c in cols if "Objetivos específicos" in c and str(obj_n) in c]
        maybe2 = [c for c in cols if "Actividades Objetivo" in c and str(obj_n) in c]
        if maybe and maybe2:
            return obj_n
    raise ValueError("No se pudo inferir el número de objetivo a partir de las columnas.")

def parse_uploaded_files(uploaded_files) -> Dict[int, pd.DataFrame]:
    """Devuelve un dict {1..6: df} estandarizando columnas y limpiando valores."""
    out = {}
    for f in uploaded_files:
        df = _read_csv_flexible(f)
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].apply(_clean_val)
        obj_n = _detect_objetivo_from_columns(list(df.columns))
        out[obj_n] = df
    return out

def _classify_row(obj_n, row):
    col_obj = f"Objetivos específicos {obj_n}"
    col_act = f"Actividades Objetivo {obj_n}"
    val_obj = _clean_val(row.get(col_obj, np.nan))
    val_act = _clean_val(row.get(col_act, np.nan))

    has_activity = isinstance(val_act, str) and len(val_act.strip()) > 1 and val_act.strip() != "-"
    code = None
    if isinstance(val_obj, str):
        m = re.match(rf"^\s*{obj_n}\.(\d+)", val_obj)
        if m:
            code = f"{obj_n}.{m.group(1)}"

    if not has_activity:
        label = "Sin actividad (fila vacía o marcador)"
    else:
        if code is not None:
            label = "Consistente con PEI (objetivo específico identificado)"
        else:
            if isinstance(val_obj, str) and str(obj_n) in val_obj:
                label = "Parcial (menciona el objetivo sin código)"
            else:
                label = "Sin correspondencia con el PEI (no se identifica objetivo específico)"
    return label, code, has_activity

def build_all_acts_and_summaries(raw_by_obj: Dict[int, pd.DataFrame]):
    """Construye el detalle de actividades y los cuadros de resumen."""
    records = []
    for obj_n, df in raw_by_obj.items():
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        # localizar columna de unidad (con o sin espacios iniciales)
        unidad_col = "Unidad Académica o Administrativa"
        if unidad_col not in df.columns:
            candidates = [c for c in df.columns if "Unidad Académica" in c]
            if candidates:
                unidad_col = candidates[0]

        for _, row in df.iterrows():
            label, code, has_activity = _classify_row(obj_n, row)
            if not has_activity:
                continue
            records.append({
                "Objetivo": obj_n,
                "AÑO": row.get("AÑO"),
                "Objetivo específico (texto)": row.get(f"Objetivos específicos {obj_n}", np.nan),
                "Código detectado": code,
                "Actividad": row.get(f"Actividades Objetivo {obj_n}", np.nan),
                "Detalle": row.get(f"Detalle de la Actividad Objetivo {obj_n}", np.nan),
                "Unidad": row.get(unidad_col, np.nan),
                "Clasificación": label,
            })

    all_acts = pd.DataFrame.from_records(records)

    # Resúmenes
    summary_obj = (
        all_acts
        .groupby(["Objetivo", "Clasificación"])
        .size()
        .reset_index(name="Cantidad")
    )
    totals = all_acts.groupby("Objetivo").size().reset_index(name="Total actividades")
    summary = summary_obj.merge(totals, on="Objetivo", how="left")
    summary["%"] = (summary["Cantidad"] / summary["Total actividades"] * 100).round(1)

    pivot_summary = summary.pivot_table(index="Objetivo", columns="Clasificación", values="%", fill_value=0).reset_index()
    pivot_counts = summary.pivot_table(index="Objetivo", columns="Clasificación", values="Cantidad", fill_value=0).reset_index()

    # Porcentaje consistente por objetivo -> columna por fila
    consist_pct = (
        summary[summary["Clasificación"] == "Consistente con PEI (objetivo específico identificado)"]
        .set_index("Objetivo")["%"]
        .to_dict()
    )
    all_acts["Consistencia del objetivo (%)"] = all_acts["Objetivo"].map(consist_pct)

    # Distribución de códigos
    dist_codigo = (
        all_acts[all_acts["Código detectado"].notna()]
        .groupby(["Objetivo", "Código detectado"])
        .size()
        .reset_index(name="N actividades")
        .sort_values(["Objetivo", "N actividades"], ascending=[True, False])
    )
    top_codigos = dist_codigo.groupby("Objetivo").head(5).reset_index(drop=True)

    # Unidades con mayor desvío
    desvio = all_acts[all_acts["Clasificación"] != "Consistente con PEI (objetivo específico identificado)"]
    unidades_desvio = (
        desvio.groupby("Unidad")
        .size()
        .reset_index(name="Desvíos (N)")
        .sort_values("Desvíos (N)", ascending=False)
    )

    return all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio

def build_excel_bytes(all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio) -> bytes:
    """Devuelve un XLSX en bytes con todas las pestañas."""
    import xlsxwriter  # noqa: F401
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        all_acts.to_excel(writer, index=False, sheet_name="Detalle actividades")
        pivot_summary.to_excel(writer, index=False, sheet_name="Porcentaje por objetivo")
        pivot_counts.to_excel(writer, index=False, sheet_name="Cantidades por objetivo")
        dist_codigo.to_excel(writer, index=False, sheet_name="Distribución por código")
        top_codigos.to_excel(writer, index=False, sheet_name="Top códigos por objetivo")
        unidades_desvio.to_excel(writer, index=False, sheet_name="Unidades con mayor desvío")
    bio.seek(0)
    return bio.read()

def build_word_bytes(all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio, matplotlib_figure) -> bytes:
    """Crea un informe .docx robusto y lo devuelve en bytes.
    Nota: importamos python-docx dentro de la función para evitar fallos de import en el arranque.
    """
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Título
    title = doc.add_paragraph("Informe de Consistencia con el PEI 2023–2027 (UCCuyo)")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.runs[0]
    run.bold = True
    run.font.size = Pt(16)

    import pandas as pd  # asegurar scope local
    doc.add_paragraph("Fecha de elaboración: ").add_run(pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")).bold = True

    # Metodología
    doc.add_heading("1. Metodología", level=1)
    doc.add_paragraph(
        "Se analizaron las actividades reportadas para los Objetivos 1 a 6. Para cada fila con actividad se verificó "
        "si el campo ‘Objetivos específicos’ inicia con el código oficial del PEI (p. ej., ‘1.5’, ‘2.3’). "
        "Clasificación: (a) Consistente con PEI (objetivo específico identificado); (b) Parcial (menciona el objetivo sin código); "
        "(c) Sin correspondencia (no se identifica objetivo específico). Se calcularon porcentajes por objetivo y se identificaron "
        "los códigos más frecuentes y las unidades con mayor número de desvíos."
    )

    # Resultados globales
    doc.add_heading("2. Resultados globales", level=1)

    # Ranking por consistencia
    consistent_label = "Consistente con PEI (objetivo específico identificado)"
    if consistent_label in pivot_summary.columns:
        consistent = pivot_summary[["Objetivo", consistent_label]].copy().sort_values(consistent_label, ascending=False)
        rank_map = {int(r.Objetivo): i+1 for i, r in consistent.reset_index(drop=True).iterrows()}
    else:
        rank_map = {}

    # Totales
    totals = all_acts.groupby("Objetivo").size().reset_index(name="Total actividades")
    total_map = dict(zip(totals["Objetivo"], totals["Total actividades"]))

    # Tabla resumen
    tbl = doc.add_table(rows=1, cols=6)
    hdr = tbl.rows[0].cells
    hdr[0].text = "Objetivo"
    hdr[1].text = "% Consistente"
    hdr[2].text = "% Parcial"
    hdr[3].text = "% Sin correspondencia"
    hdr[4].text = "Total act."
    hdr[5].text = "Consistencia (ranking)"

    def _get_pct(pivot, obj, label):
        row = pivot[pivot["Objetivo"]==obj]
        if row.empty: return 0.0
        return float(row[label].values[0]) if label in row.columns else 0.0

    for obj in sorted(pivot_summary["Objetivo"].tolist()):
        r = tbl.add_row().cells
        r[0].text = str(obj)
        r[1].text = f"{_get_pct(pivot_summary, obj, 'Consistente con PEI (objetivo específico identificado)'):.1f}%"
        r[2].text = f"{_get_pct(pivot_summary, obj, 'Parcial (menciona el objetivo sin código)'):.1f}%"
        r[3].text = f"{_get_pct(pivot_summary, obj, 'Sin correspondencia con el PEI (no se identifica objetivo específico)'):.1f}%"
        r[4].text = str(int(total_map.get(obj, 0)))
        r[5].text = f"{rank_map.get(int(obj), '')}º"

    # Insertar gráfico desde figura matplotlib ya creada
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        chart_path = os.path.join(tmpdir, "consistencia.png")
        matplotlib_figure.savefig(chart_path, dpi=200, bbox_inches="tight")
        doc.add_paragraph()
        doc.add_picture(chart_path, width=Inches(6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Análisis por objetivo
    doc.add_heading("3. Análisis por objetivo", level=1)
    for obj in sorted(pivot_summary["Objetivo"].tolist()):
        doc.add_heading(f"Objetivo {obj}", level=2)
        pct_cons = _get_pct(pivot_summary, obj, "Consistente con PEI (objetivo específico identificado)")
        pct_parc = _get_pct(pivot_summary, obj, "Parcial (menciona el objetivo sin código)")
        pct_sin  = _get_pct(pivot_summary, obj, "Sin correspondencia con el PEI (no se identifica objetivo específico)")
        tot = int(total_map.get(obj, 0))
        doc.add_paragraph(
            f"Total de actividades analizadas: {tot}. Consistencia: {pct_cons:.1f}%. "
            f"Parciales: {pct_parc:.1f}%. Sin correspondencia: {pct_sin:.1f}%."
        )
        tc = top_codigos[top_codigos["Objetivo"]==obj].copy()
        if not tc.empty:
            doc.add_paragraph("Códigos específicos más frecuentes:")
            for _, r in tc.iterrows():
                doc.add_paragraph(f"• {r['Código detectado']}: {int(r['N actividades'])} actividades")

        sub = all_acts[(all_acts["Objetivo"]==obj) & (all_acts["Clasificación"]!="Consistente con PEI (objetivo específico identificado)")]
        if not sub.empty:
            doc.add_paragraph("Ejemplos de actividades para revisión (desvíos/parciales):")
            for _, r in sub.head(5).iterrows():
                doc.add_paragraph(f"— {r['Actividad']}")

    # Unidades con mayor desvío
    doc.add_heading("4. Unidades con mayor número de desvíos", level=1)
    if not unidades_desvio.empty:
        t2 = doc.add_table(rows=1, cols=2)
        t2.rows[0].cells[0].text = "Unidad Académica/Administrativa"
        t2.rows[0].cells[1].text = "Desvíos (N)"
        for _, rr in unidades_desvio.head(10).iterrows():
            rw = t2.add_row().cells
            rw[0].text = str(rr["Unidad"])
            rw[1].text = str(int(rr["Desvíos (N)"]))
    else:
        doc.add_paragraph("No se detectaron desvíos.")

    # Conclusiones
    doc.add_heading("5. Conclusiones y recomendaciones", level=1)
    doc.add_paragraph(
        "Los porcentajes de consistencia por objetivo muestran el grado de alineación formal de las actividades con los objetivos "
        "específicos del PEI. Un porcentaje alto indica buena trazabilidad entre planificación y ejecución. Sin embargo, los ítems clasificados "
        "como parciales o sin correspondencia sugieren oportunidades para mejorar la calidad del registro y la vinculación explícita con el código del PEI."
    )
    doc.add_paragraph(
        "Recomendaciones: (i) reforzar la carga con validación que obligue a seleccionar el código del objetivo específico; "
        "(ii) incluir descripciones breves normalizadas que referencien el epígrafe del PEI; "
        "(iii) realizar revisiones periódicas por unidad para detectar patrones de desvío; "
        "(iv) retroalimentar a responsables de carga con ejemplos de buenas prácticas; "
        "(v) evaluar, en la próxima iteración, la correspondencia semántica entre actividad y epígrafe mediante un cotejo automatizado."
    )

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()
