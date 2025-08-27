from io import BytesIO
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm

# -----------------------------
# Limpieza y utilidades básicas
# -----------------------------

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas: minúsculas, sin espacios dobles, sin tildes básicas."""
    import unicodedata, re
    def norm(s):
        s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
        s = s.lower().strip()
        s = re.sub(r'\s+', ' ', s)
        return s
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

def guess_consistency_column(df: pd.DataFrame):
    """Heurística para adivinar la columna que trae la consistencia."""
    candidates = [
        "consistencia", "tipo consistencia", "clasificacion", "categoria",
        "estado", "resultado", "correspondencia", "coherencia", "nivel",
    ]
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # Por substrings
    for c in df.columns:
        if any(key in c for key in ["consist", "coheren", "correspon", "clasif", "categoria", "estado", "resultado", "nivel"]):
            return c
    return None

# ---------------------------------------
# Cálculo de resumen de consistencias
# ---------------------------------------

def compute_consistency_summary(df: pd.DataFrame, colname, label_map):
    """
    colname: nombre de la columna que clasifica (o None si no existe)
    label_map: dict con claves 'plena', 'parcial', 'nula' -> listas de etiquetas (lowercase)
    """
    out = {"total": len(df), "plena": 0, "parcial": 0, "nula": 0, "sin_clasificar": 0}

    table = df.copy()
    if colname is None or colname not in df.columns:
        # Sin columna: todo sin clasificar
        out["sin_clasificar"] = len(df)
        table["clasificacion_calculada"] = "sin clasificar"
        return {**out, "table": table}

    def classify(val):
        s = str(val).strip().lower()
        if any(k in s for k in label_map["plena"]):
            return "plena"
        if any(k in s for k in label_map["parcial"]):
            return "parcial"
        if any(k in s for k in label_map["nula"]):
            return "nula"
        return "sin clasificar"

    table["clasificacion_calculada"] = df[colname].map(classify)
    counts = table["clasificacion_calculada"].value_counts(dropna=False).to_dict()

    out["plena"] = counts.get("plena", 0)
    out["parcial"] = counts.get("parcial", 0)
    out["nula"] = counts.get("nula", 0)
    out["sin_clasificar"] = counts.get("sin clasificar", 0)

    return {**out, "table": table}

# ---------------------------------------
# Reportes
# ---------------------------------------

def build_excel_report(summary_df: pd.DataFrame, detail_tables):
    """Devuelve bytes de un Excel con hoja 'Resumen' y hojas de detalle por archivo."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Resumen")
        for name, table in detail_tables:
            # Hoja segura
            sheet = (name[:30] or "Detalle").replace("/", "-")
            table.to_excel(writer, index=False, sheet_name=sheet)
    bio.seek(0)
    return bio.getvalue()

def build_word_report(summary_df: pd.DataFrame, totals, names):
    """Crea un .docx narrado con hallazgos principales."""
    doc = Document()

    # Estilo simple
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    doc.add_heading("Informe de coherencia – PEI UCCuyo 2023–2027", level=1)
    doc.add_paragraph(f"Fecha de generación: {pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')}")

    # Totales
    doc.add_heading("Resumen ejecutivo", level=2)
    p = doc.add_paragraph()
    p.add_run("Totales consolidados: ").bold = True
    p.add_run(
        f"Actividades={int(totals['Total actividades'])}, "
        f"Plena={int(totals['Consistencia plena'])}, "
        f"Parcial={int(totals['Consistencia parcial'])}, "
        f"Nula={int(totals['Consistencia nula'])}, "
        f"Sin clasificar={int(totals['Sin clasificar'])}."
    )

    # Tabla resumen
    doc.add_heading("Tabla resumen por objetivo (archivo)", level=2)
    table = doc.add_table(rows=1, cols=len(summary_df.columns))
    hdr = table.rows[0].cells
    for j, col in enumerate(summary_df.columns):
        hdr[j].text = str(col)

    for _, row in summary_df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(summary_df.columns):
            cells[j].text = str(row[col])

    # Comentarios automáticos simples
    doc.add_heading("Comentarios automáticos", level=2)
    if int(totals.get("Consistencia nula", 0)) > 0:
        doc.add_paragraph("• Se observan actividades con consistencia nula. Se recomienda revisar definiciones y criterios de mapeo o realinear las actividades con las acciones del PEI.")
    if int(totals.get("Sin clasificar", 0)) > 0:
        doc.add_paragraph("• Existen actividades sin clasificar. Verifique que la columna de consistencia y las etiquetas estén correctamente configuradas para cada archivo.")
    if int(totals.get("Consistencia plena", 0)) >= int(totals.get("Consistencia parcial", 0)):
        doc.add_paragraph("• Predominan las actividades con consistencia plena/parcial, lo cual sugiere una alineación razonable con los objetivos del PEI.")

    # Pie
    doc.add_paragraph("—")
    doc.add_paragraph("Generado automáticamente por la Calculadora de Consistencia – Secretaría de Investigación UCCuyo.")

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
