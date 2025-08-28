from io import BytesIO
from pathlib import Path
import re
import unicodedata
from typing import List, Dict, Any, Tuple
import pandas as pd
from rapidfuzz import fuzz
from pypdf import PdfReader

# =============================
# Normalización y helpers
# =============================

SPANISH_STOP = set("""a al algo alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde dos el la los las en entre era erais eramos eran es esa esas ese eso esos esta estas este esto estos fue fuerais fuéramos fueran fui fuimos ha haber habia había habiais habíamos habían han has hasta hay la lo las los le les mas más me mi mis mucho muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque que quien quienes se sin sobre sois somos son soy su sus te tenia tenía teniais teníamos tenían tengo ti tu tus un una uno unas unos y ya""".split())

def strip_accents(s: str) -> str:
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

def normalize_text(s: str) -> str:
    s = strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9áéíóúñ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in SPANISH_STOP and len(t) > 2]
    return " ".join(tokens)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        s = strip_accents(str(s)).lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out

# =============================
# Parseo de PEI (PDF)
# =============================

def parse_pei_pdf(file_like) -> Dict[str, Any]:
    """
    Devuelve estructura con objetivos -> especificos -> acciones/indicadores
    """
    reader = PdfReader(file_like)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text_norm = strip_accents(text)
    lines = [l.strip() for l in text_norm.splitlines() if l.strip()]
    obj_map: Dict[str, Dict[str, Any]] = {}
    current_obj = None
    current_spec = None
    mode = None  # "acciones" | "indicadores" | None

    # patrones
    re_obj = re.compile(r"^OBJETIVO\s+(\d)\s*[:\-]?", re.IGNORECASE)
    re_spec = re.compile(r"^(\d\.\d)\.?")  # 1.1, 2.3, ...
    for raw in lines:
        l = raw
        m = re_obj.search(l)
        if m:
            current_obj = m.group(1)
            obj_map.setdefault(current_obj, {"titulo": "", "especificos": {}})
            current_spec = None
            mode = None
            continue
        m2 = re_spec.match(l)
        if m2 and current_obj:
            current_spec = m2.group(1)
            obj_map[current_obj]["especificos"].setdefault(current_spec, {"titulo": l, "acciones": [], "indicadores": []})
            mode = None
            continue
        # detectores de secciones
        if current_obj and current_spec:
            low = l.lower()
            if "acciones" == low.strip():
                mode = "acciones"; continue
            if "indicadores" == low.strip():
                mode = "indicadores"; continue
            if low.startswith("responsable") or low.startswith("plazos"):
                mode = None; continue
            # contenido
            if mode == "acciones":
                if re.match(r"^\d+\.\d+\.\d+\.", l) or low.startswith("-") or low.startswith("\u2212"):
                    obj_map[current_obj]["especificos"][current_spec]["acciones"].append(l)
            elif mode == "indicadores":
                if not re.match(r"^OBJETIVO", l, re.IGNORECASE) and not re_spec.match(l):
                    obj_map[current_obj]["especificos"][current_spec]["indicadores"].append(l)
    return obj_map

def build_plan_index(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convierte el dict del PEI en una lista de entradas buscables.
    Cada entrada representa un conjunto (objetivo, específico, acción/indicadores)
    """
    index = []
    for obj, objdata in plan.items():
        especs = objdata.get("especificos", {})
        for spec, sdata in especs.items():
            acciones = sdata.get("acciones", []) or []
            indicadores = sdata.get("indicadores", []) or []
            if not acciones and not indicadores:
                index.append({
                    "objetivo": obj,
                    "especifico": spec,
                    "texto": normalize_text(sdata.get("titulo","")),
                    "tipo": "titulo"
                })
            else:
                for a in (acciones or [""]):
                    text = " ".join([a] + indicadores)
                    index.append({
                        "objetivo": obj,
                        "especifico": spec,
                        "texto": normalize_text(text),
                        "tipo": "accion_indicadores",
                        "accion": a,
                        "indicadores": indicadores
                    })
    return [e for e in index if e.get("texto")]

# =============================
# Matching actividad ↔ PEI
# =============================

def build_activity_text(row: pd.Series) -> str:
    candidates = [
        "actividad", "accion", "acciones", "descripcion", "descripción",
        "detalle", "indicador", "indicadores", "resultado", "resultados",
        "objetivo", "objetivos", "titulo", "titulo actividad", "nombre"
    ]
    cols = [c for c in row.index if c in candidates]
    if not cols:
        cols = list(row.index)
    parts = []
    for c in cols:
        v = str(row[c])
        if v and v.lower() not in ("nan","none"):
            parts.append(v)
    return " ".join(parts)

def match_activity_to_plan(activity_text: str, index: List[Dict[str, Any]], objective_hint: str=None):
    q = normalize_text(activity_text)
    best = None
    best_score = -1.0
    for entry in index:
        s = fuzz.token_set_ratio(q, entry["texto"])
        if objective_hint and objective_hint == entry["objetivo"]:
            s = s + 5
        if s > best_score:
            best = entry; best_score = s
    return best, float(best_score)

def classify_consistency(score: float, ok_objective: bool, thresholds: Dict[str, float]) -> str:
    t_plena = thresholds.get("plena", 88.0)
    t_parcial = thresholds.get("parcial", 68.0)
    if score >= t_plena and ok_objective:
        return "plena"
    if score >= t_parcial:
        return "parcial"
    return "nula"

# =============================
# Resúmenes y reportes
# =============================

def compute_consistency_with_plan(dfs: List[pd.DataFrame], names: List[str], plan_index: List[Dict[str,Any]], thresholds: Dict[str,float]):
    detail_tables = []
    summary_rows = []
    matrix_rows = []

    for name, df in zip(names, dfs):
        df = df.copy()
        if not len(df):
            continue
        m = re.match(r"^\s*([1-6])", strip_accents(name))
        objective_hint = m.group(1) if m else None

        best_objs = []
        best_specs = []
        best_scores = []
        best_acciones = []
        best_indics = []

        for _, row in df.iterrows():
            text = build_activity_text(row)
            match, score = match_activity_to_plan(text, plan_index, objective_hint=objective_hint)
            if match is None:
                best_objs.append(None); best_specs.append(None); best_scores.append(0); best_acciones.append(""); best_indics.append([])
                continue
            ok_objective = (objective_hint is None) or (match["objetivo"] == objective_hint)
            cat = classify_consistency(score, ok_objective, thresholds)
            best_objs.append(match["objetivo"])
            best_specs.append(match["especifico"])
            best_scores.append(score)
            best_acciones.append(match.get("accion",""))
            best_indics.append(", ".join(match.get("indicadores",[])))

            # guardar categoría
        df["pei_objetivo"] = best_objs
        df["pei_especifico"] = best_specs
        df["pei_accion"] = best_acciones
        df["pei_indicadores"] = best_indics
        df["pei_score"] = best_scores

        ok_obj = [ (str(o).strip() != "None") and ((objective_hint is None) or (str(o)==objective_hint)) for o in df["pei_objetivo"] ]
        df["clasificacion_calculada"] = [
            classify_consistency(s, ok, thresholds) for s, ok in zip(df["pei_score"], ok_obj)
        ]

        counts = df["clasificacion_calculada"].value_counts(dropna=False).to_dict()
        total = len(df)
        summary_rows.append({
            "Objetivo (archivo)": name,
            "Total actividades": total,
            "Consistencia plena": int(counts.get("plena",0)),
            "Consistencia parcial": int(counts.get("parcial",0)),
            "Consistencia nula": int(counts.get("nula",0)),
            "Sin clasificar": int(counts.get("sin clasificar",0))
        })

        detail_tables.append((name, df))

        mat = df.groupby(["pei_objetivo","clasificacion_calculada"]).size().unstack(fill_value=0)
        mat["archivo"] = name
        mat["pei_objetivo"] = mat.index
        matrix_rows.append(mat.reset_index(drop=True))

    summary_df = pd.DataFrame(summary_rows)
    matrix_df = pd.concat(matrix_rows, ignore_index=True) if matrix_rows else pd.DataFrame()

    return summary_df, detail_tables, matrix_df

def build_excel_report(summary_df: pd.DataFrame, detail_tables, labels_used=None, matrix_df: pd.DataFrame=None):
    from pandas import ExcelWriter
    bio = BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        d = summary_df.copy()
        if not d.empty:
            for col in ["Consistencia plena","Consistencia parcial","Consistencia nula","Sin clasificar"]:
                d[f"% {col}"] = (d[col] / d["Total actividades"]).round(4)
        d.to_excel(writer, index=False, sheet_name="Resumen")

        if isinstance(matrix_df, pd.DataFrame) and not matrix_df.empty:
            matrix_df.to_excel(writer, index=False, sheet_name="Matriz_Objetivos")

        for name, table in detail_tables:
            sheet = (name[:30] or "Detalle").replace("/", "-")
            table.to_excel(writer, index=False, sheet_name=sheet)

        if labels_used:
            pd.DataFrame.from_dict(labels_used, orient="index").to_excel(writer, sheet_name="Etiquetas", header=False)
    bio.seek(0)
    return bio.getvalue()

def build_word_report(summary_df: pd.DataFrame, totals, names):
    try:
        from docx import Document
        from docx.shared import Pt
    except Exception as e:
        raise ImportError("python-docx no está instalado") from e

    def pct(a, b): return 0.0 if b==0 else round(100*a/b,2)

    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Times New Roman"; style.font.size = Pt(11)

    doc.add_heading("Informe de correlación con PEI (Automático)", level=1)
    if not summary_df.empty:
        T = int(summary_df["Total actividades"].sum())
        P = int(summary_df["Consistencia plena"].sum())
        Pa = int(summary_df["Consistencia parcial"].sum())
        N = int(summary_df["Consistencia nula"].sum())
        S = int(summary_df["Sin clasificar"].sum())
    else:
        T=P=Pa=N=S=0

    doc.add_paragraph(f"Total de actividades analizadas: {T}. Plena: {P} ({pct(P,T)}%), Parcial: {Pa} ({pct(Pa,T)}%), Nula: {N} ({pct(N,T)}%), Sin clasificar: {S} ({pct(S,T)}%).")
    doc.add_paragraph("Criterio: clasificación conservadora basada en similitud con Acciones/Indicadores del PEI (PDF) y coincidencia de objetivo cuando está disponible.")

    if not summary_df.empty:
        table = doc.add_table(rows=1, cols=len(summary_df.columns))
        for j, c in enumerate(summary_df.columns): table.rows[0].cells[j].text = str(c)
        for _, row in summary_df.iterrows():
            cells = table.add_row().cells
            for j, c in enumerate(summary_df.columns): cells[j].text = str(row[c])

    bio = BytesIO(); doc.save(bio); bio.seek(0)
    return bio.getvalue()