
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

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.dropna(how="all", axis=1)
    def row_empty(r):
        for v in r:
            s = str(v).strip().lower()
            if s not in ("", "nan", "none", "-"):
                return False
        return True
    mask = tmp.apply(row_empty, axis=1)
    return tmp.loc[~mask].reset_index(drop=True)

# =============================
# Parseo de PEI (PDF) — modo A
# =============================

def parse_pei_pdf(file_like) -> Dict[str, Any]:
    reader = PdfReader(file_like)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text_norm = strip_accents(text)
    lines = [l.strip() for l in text_norm.splitlines() if l.strip()]
    obj_map: Dict[str, Dict[str, Any]] = {}
    current_obj = None
    current_spec = None
    mode = None

    re_obj = re.compile(r"^OBJETIVO\s+(\d)\s*[:\-]?", re.IGNORECASE)
    re_spec = re.compile(r"^(\d\.\d)\.?")
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
        if current_obj and current_spec:
            low = l.lower()
            if "acciones" == low.strip():
                mode = "acciones"; continue
            if "indicadores" == low.strip():
                mode = "indicadores"; continue
            if low.startswith("responsable") or low.startswith("plazos"):
                mode = None; continue
            if mode == "acciones":
                if re.match(r"^\d+\.\d+\.\d+\.", l) or low.startswith("-") or low.startswith("\u2212"):
                    obj_map[current_obj]["especificos"][current_spec]["acciones"].append(l)
            elif mode == "indicadores":
                if not re.match(r"^OBJETIVO", l, re.IGNORECASE) and not re_spec.match(l):
                    obj_map[current_obj]["especificos"][current_spec]["indicadores"].append(l)
    return obj_map

def build_plan_index(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
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

def compute_consistency_with_plan(dfs: List[pd.DataFrame], names: List[str], plan_index: List[Dict[str,Any]], thresholds: Dict[str,float]):
    detail_tables, summary_rows, matrix_rows = [], [], []
    for name, df in zip(names, dfs):
        df = clean_rows(df.copy())
        if not len(df): 
            continue
        m = re.match(r"^\s*([1-6])", strip_accents(name))
        objective_hint = m.group(1) if m else None

        best_objs, best_specs, best_scores, best_acciones, best_indics, categories = [], [], [], [], [], []
        for _, row in df.iterrows():
            text = " ".join(str(x) for x in row.values if str(x).strip())
            match, score = match_activity_to_plan(text, plan_index, objective_hint=objective_hint)
            if match is None:
                best_objs.append(None); best_specs.append(None); best_scores.append(0); best_acciones.append(""); best_indics.append(""); categories.append("nula"); continue
            ok_objective = (objective_hint is None) or (match["objetivo"] == objective_hint)
            cat = classify_consistency(score, ok_objective, thresholds)
            categories.append(cat)
            best_objs.append(match["objetivo"]); best_specs.append(match["especifico"]); best_scores.append(score)
            best_acciones.append(match.get("accion","")); best_indics.append(", ".join(match.get("indicadores",[])))

        df["pei_objetivo"] = best_objs
        df["pei_especifico"] = best_specs
        df["pei_accion"] = best_acciones
        df["pei_indicadores"] = best_indics
        df["pei_score"] = best_scores
        df["clasificacion_calculada"] = categories

        counts = df["clasificacion_calculada"].value_counts(dropna=False).to_dict()
        total = int(len(df))
        summary_rows.append({
            "Objetivo (archivo)": name,
            "Total actividades": total,
            "Consistencia plena": int(counts.get("plena",0)),
            "Consistencia parcial": int(counts.get("parcial",0)),
            "Consistencia nula": int(counts.get("nula",0))
        })
        detail_tables.append((name, df))

        mat = df.groupby(["pei_objetivo","clasificacion_calculada"]).size().unstack(fill_value=0)
        mat["archivo"] = name
        mat["pei_objetivo"] = mat.index
        matrix_rows.append(mat.reset_index(drop=True))

    summary_df = pd.DataFrame(summary_rows)
    matrix_df = pd.concat(matrix_rows, ignore_index=True) if matrix_rows else pd.DataFrame()
    return summary_df, detail_tables, matrix_df

# =============================
# Modo B: correlación CSV col2↔col1
# =============================

def compute_pairwise_consistency(dfs: List[pd.DataFrame], names: List[str], columns: List[Tuple[str,str]], thresholds: Dict[str,float]):
    detail_tables, summary_rows = [], []
    for name, df, (col_obj, col_act) in zip(names, dfs, columns):
        df = clean_rows(df.copy())
        if not len(df): 
            continue
        # si vienen por índice, convierto a nombre
        if isinstance(col_obj, int): col_obj = df.columns[col_obj]
        if isinstance(col_act, int): col_act = df.columns[col_act]

        scores, cats = [], []
        for _, row in df.iterrows():
            a = normalize_text(row.get(col_obj, ""))
            b = normalize_text(row.get(col_act, ""))
            s = fuzz.token_set_ratio(a, b)
            ok_obj = True  # en este modo no exigimos objetivo externo
            # umbrales más duros por defecto (equivalentes a modo A)
            cat = classify_consistency(float(s), ok_obj, thresholds)
            scores.append(float(s)); cats.append(cat)
        df["score_obj_vs_act"] = scores
        df["clasificacion_calculada"] = cats
        df["col_objetivo"] = col_obj
        df["col_actividad"] = col_act

        counts = df["clasificacion_calculada"].value_counts(dropna=False).to_dict()
        total = int(len(df))
        summary_rows.append({
            "Objetivo (archivo)": name,
            "Total actividades": total,
            "Consistencia plena": int(counts.get("plena",0)),
            "Consistencia parcial": int(counts.get("parcial",0)),
            "Consistencia nula": int(counts.get("nula",0))
        })
        detail_tables.append((name, df))

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail_tables

# =============================
# Reportes
# =============================

def build_excel_report(summary_df: pd.DataFrame, detail_tables, labels_used=None, matrix_df: pd.DataFrame=None):
    from pandas import ExcelWriter
    bio = BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        # Resumen (solo conteos)
        summary_df.to_excel(writer, index=False, sheet_name="Resumen")

        # Porcentajes
        if not summary_df.empty:
            p = summary_df.copy()
            for col in ["Consistencia plena","Consistencia parcial","Consistencia nula"]:
                p[f"% {col}"] = (p[col] / p["Total actividades"]).round(4)
            p = p[["Objetivo (archivo)", "% Consistencia plena", "% Consistencia parcial", "% Consistencia nula"]]
            T = float(summary_df["Total actividades"].sum())
            if T > 0:
                g = pd.DataFrame([{
                    "Objetivo (archivo)": "TOTAL GLOBAL",
                    "% Consistencia plena": round(summary_df["Consistencia plena"].sum()/T,4),
                    "% Consistencia parcial": round(summary_df["Consistencia parcial"].sum()/T,4),
                    "% Consistencia nula": round(summary_df["Consistencia nula"].sum()/T,4),
                }])
                p = pd.concat([p, g], ignore_index=True)
            p.to_excel(writer, index=False, sheet_name="Porcentajes")

        # Matriz (solo aplica en modo A)
        if isinstance(matrix_df, pd.DataFrame) and not matrix_df.empty:
            matrix_df.to_excel(writer, index=False, sheet_name="Matriz_Objetivos")

        # Detalle
        for name, table in detail_tables:
            sheet = (name[:30] or "Detalle").replace("/", "-")
            table.to_excel(writer, index=False, sheet_name=sheet)

    bio.seek(0)
    return bio.getvalue()

def build_word_report(summary_df: pd.DataFrame, totals, names):
    try:
        from docx import Document
        from docx.shared import Pt
    except Exception as e:
        raise ImportError("python-docx no está instalado") from e

    def pct(a, b): return 0.0 if b==0 else round(100*a/b,2)

    T = int(summary_df["Total actividades"].sum()) if not summary_df.empty else 0
    P = int(summary_df["Consistencia plena"].sum()) if not summary_df.empty else 0
    Pa = int(summary_df["Consistencia parcial"].sum()) if not summary_df.empty else 0
    N = int(summary_df["Consistencia nula"].sum()) if not summary_df.empty else 0

    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Times New Roman"; style.font.size = Pt(11)

    doc.add_heading("Informe de correlación automática", level=1)
    doc.add_paragraph(f"Total de actividades: {T}. Plena: {P} ({pct(P,T)}%). Parcial: {Pa} ({pct(Pa,T)}%). Nula: {N} ({pct(N,T)}%).")

    if not summary_df.empty:
        cols = ["Objetivo (archivo)","Total actividades","Consistencia plena","Consistencia parcial","Consistencia nula"]
        table = doc.add_table(rows=1, cols=len(cols))
        for j, c in enumerate(cols): table.rows[0].cells[j].text = c
        for _, row in summary_df[cols].iterrows():
            cells = table.add_row().cells
            for j, c in enumerate(cols): cells[j].text = str(row[c])

    bio = BytesIO(); doc.save(bio); bio.seek(0)
    return bio.getvalue()
