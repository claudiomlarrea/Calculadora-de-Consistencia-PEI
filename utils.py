
from io import BytesIO
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Evitar ImportError si no está pypdf instalado (no siempre usamos PDF)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # se maneja dentro de la función

# -------------------- Normalización --------------------
EMPTY_TOKENS = {"", "nan", "none", "-", "—", "–"}
SPANISH_STOP = set("""a al algo alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde dos el la los las en entre era erais eramos eran es esa esas ese eso esos esta estas este esto estos fue fuerais fuéramos fueran fui fuimos ha haber habia había habiais habíamos habían han has hasta hay la lo las los le les mas más me mi mis mucho muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque que quien quienes se sin sobre sois somos son soy su sus te tenia tenía teniais teníamos tenían tengo ti tu tus un una uno unas unos y ya""".split())

def strip_accents(s: str) -> str:
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')

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

def is_empty_value(v) -> bool:
    return str(v).strip().lower() in EMPTY_TOKENS

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.dropna(how="all", axis=1)
    def row_empty(r):
        for v in r:
            s = str(v).strip().lower()
            if s not in EMPTY_TOKENS:
                return False
        return True
    mask = tmp.apply(row_empty, axis=1)
    return tmp.loc[~mask].reset_index(drop=True)

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    obj_candidates = [c for c in cols if any(k in c for k in [
        "objetivo especific", "objetivos especific", "objetivo espe", "objetivo 1", "obj 1", "especifico"
    ])]
    act_candidates = [c for c in cols if any(k in c for k in [
        "actividades objetivo", "actividad objetivo", "actividad", "acciones", "accion"
    ])]
    obj_col = obj_candidates[0] if obj_candidates else (cols[0] if cols else None)
    act_candidates = [c for c in act_candidates if c != obj_col] or [c for c in cols if c != obj_col]
    act_col = act_candidates[0] if act_candidates else None
    return obj_col, act_col

def count_valid_pairs(df: pd.DataFrame, col_obj: str, col_act: str) -> int:
    def ok(v): return not is_empty_value(v)
    return int(((df[col_obj].apply(ok)) & (df[col_act].apply(ok))).sum())

# -------------------- Consistencia Independiente (col 2 vs col 1) --------------------
def classify(score: float, thr_full=88.0, thr_partial=68.0) -> str:
    if score >= thr_full:
        return "plena"
    if score >= thr_partial:
        return "parcial"
    return "nula"

def best_alt_objective(activity_norm: str, current_obj: str, unique_objs_norm):
    """Mejor objetivo alternativo (texto y score) para la actividad (excluye el objetivo actual)."""
    best_txt, best_score = None, -1.0
    cur_norm = normalize_text(current_obj)
    for raw, norm in unique_objs_norm:
        if norm == cur_norm:
            continue
        s = fuzz.token_set_ratio(activity_norm, norm)
        if s > best_score:
            best_score = s; best_txt = raw
    return best_txt, float(best_score)

def analyze_independent(df: pd.DataFrame, name: str, col_obj: str, col_act: str, thr_full=88.0, thr_partial=68.0):
    """Devuelve:
       - resumen (dict)
       - detalle (DataFrame) con columnas extra pedidas
       - perf_obj (DataFrame rendimiento por objetivo)
       - mejoras (DataFrame sugerencias con Δ p.p. >= 15)
       - duplicadas (DataFrame sospecha de duplicados)
    """
    df = clean_rows(df.copy())
    valid_mask = (~df[col_obj].apply(is_empty_value)) & (~df[col_act].apply(is_empty_value))
    data = df.loc[valid_mask].reset_index(drop=True).copy()

    # normalizados
    data["_obj_norm"] = data[col_obj].astype(str).map(normalize_text)
    data["_act_norm"] = data[col_act].astype(str).map(normalize_text)

    # set de objetivos
    unique_objs = list(dict.fromkeys(data[col_obj].astype(str).tolist()))
    unique_objs_norm = [(t, normalize_text(t)) for t in unique_objs]

    # scores, categorías y sugerencias
    scores, cats, alt_objs, alt_scores, deltas = [], [], [], [], []
    for o_raw, a_raw, o, a in zip(data[col_obj], data[col_act], data["_obj_norm"], data["_act_norm"]):
        s = fuzz.token_set_ratio(o, a)
        cat = classify(float(s), thr_full, thr_partial)
        alt_txt, alt_s = best_alt_objective(a, str(o_raw), unique_objs_norm)
        scores.append(float(s)); cats.append(cat); alt_objs.append(alt_txt); alt_scores.append(float(alt_s)); deltas.append(float(alt_s - s))

    # --- columnas pedidas ---
    data["Actividad (seleccionada)"] = data[col_act].astype(str)
    data["Mejor objetivo propuesto"] = alt_objs
    data["% si se ubica en objetivo propuesto"] = [round(x,1) for x in alt_scores]

    # info de situación actual
    data["score_obj_vs_act"] = scores
    data["clasificacion"] = cats
    data["Δ p.p. (objetivo propuesto - actual)"] = [round(x,1) for x in deltas]

    # Resumen
    counts = data["clasificacion"].value_counts(dropna=False).to_dict()
    total = len(data)
    resumen = {
        "Fuente": name,
        "Total actividades": int(total),
        "Consistencia plena": int(counts.get("plena",0)),
        "Consistencia parcial": int(counts.get("parcial",0)),
        "Consistencia nula": int(counts.get("nula",0))
    }

    # Rendimiento por objetivo
    g = data.groupby(col_obj)["score_obj_vs_act"].agg(["count","mean","median"])
    g["p25"] = data.groupby(col_obj)["score_obj_vs_act"].quantile(0.25)
    g["p75"] = data.groupby(col_obj)["score_obj_vs_act"].quantile(0.75)
    g["std"] = data.groupby(col_obj)["score_obj_vs_act"].std(ddof=0)
    lvl = data.assign(_low = data["clasificacion"].eq("nula")).groupby(col_obj)["_low"].mean().rename("% en Bajo")
    perf_obj = g.join(lvl).reset_index().rename(columns={col_obj:"Objetivo específico","count":"N","mean":"Promedio","median":"Mediana"})
    perf_obj["% en Bajo"] = (perf_obj["% en Bajo"]*100).round(1)
    perf_obj = perf_obj.sort_values("% en Bajo", ascending=False)

    # Potencial de reubicación (ganancia >= 15 p.p.)
    mejoras = data.copy()
    mejoras = mejoras.loc[mejoras["Δ p.p. (objetivo propuesto - actual)"] >= 15.0].sort_values("Δ p.p. (objetivo propuesto - actual)", ascending=False)
    mejoras = mejoras[[
        "Actividad (seleccionada)",  # actividad
        col_obj,                      # objetivo actual
        "score_obj_vs_act",           # % actual
        "Mejor objetivo propuesto",   # objetivo sugerido
        "% si se ubica en objetivo propuesto",  # % sugerido
        "Δ p.p. (objetivo propuesto - actual)"
    ]].rename(columns={
        "Actividad (seleccionada)":"Actividad",
        col_obj:"Obj. actual",
        "score_obj_vs_act":"% actual",
        "Mejor objetivo propuesto":"Obj. sugerido",
        "% si se ubica en objetivo propuesto":"% sugerido",
        "Δ p.p. (objetivo propuesto - actual)":"Δ p.p."
    })
    mejoras["% actual"] = mejoras["% actual"].round(1)
    mejoras["% sugerido"] = mejoras["% sugerido"].round(1)
    mejoras["Δ p.p."] = mejoras["Δ p.p."].round(1)

    # Duplicadas
    dup = (data.assign(act_norm=data["_act_norm"])
               .groupby("act_norm").size().reset_index(name="Repeticiones"))
    dup = dup.loc[dup["Repeticiones"]>1].sort_values("Repeticiones", ascending=False)
    dup = dup.rename(columns={"act_norm":"Actividad (normalizada)"})

    return resumen, data, perf_obj, mejoras, dup

# -------------------- Exportadores --------------------
def excel_consolidado(resumen: Dict[str,int], detalle: pd.DataFrame,
                      perf_obj: pd.DataFrame, mejoras: pd.DataFrame, duplicadas: pd.DataFrame):
    """Excel tipo 'informe_consistencia_pei_consolidado' con columnas pedidas."""
    from pandas import ExcelWriter
    bio = BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
        p = pd.DataFrame([{
            "Fuente": resumen["Fuente"],
            "% Consistencia plena": round(resumen["Consistencia plena"]/max(resumen["Total actividades"],1),4),
            "% Consistencia parcial": round(resumen["Consistencia parcial"]/max(resumen["Total actividades"],1),4),
            "% Consistencia nula": round(resumen["Consistencia nula"]/max(resumen["Total actividades"],1),4),
        }])
        p.to_excel(writer, index=False, sheet_name="Porcentajes")
        perf_obj.to_excel(writer, index=False, sheet_name="Rendimiento_por_objetivo")
        mejoras.to_excel(writer, index=False, sheet_name="Potencial_reubicacion")
        duplicadas.to_excel(writer, index=False, sheet_name="Duplicadas")
        # Detalle incluye las nuevas 3 columnas pedidas
        detalle.to_excel(writer, index=False, sheet_name="Detalle")
    bio.seek(0)
    return bio.getvalue()

# -------------------- (Opcional) Parseo PEI PDF seguro --------------------
def parse_pei_pdf(file_like) -> Dict[str, Any]:
    if PdfReader is None:
        # Si no hay pypdf instalado devolvemos índice vacío y evitamos romper import
        return {}
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
