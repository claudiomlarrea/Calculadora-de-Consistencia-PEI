from io import BytesIO
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from rapidfuzz import fuzz
from pypdf import PdfReader

# -------------------- Helpers --------------------
EMPTY_TOKENS = {"", "nan", "none", "-", "–", "—"}

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
    """Versión menos restrictiva - solo considera vacíos los valores realmente nulos"""
    if pd.isna(v):
        return True
    v_str = str(v).strip().lower()
    # Solo considera vacíos: cadenas completamente vacías, "nan", "none"
    return v_str in {"", "nan", "none"}

def is_meaningful_text(v, min_length: int = 5) -> bool:
    """Verifica si el texto tiene contenido significativo"""
    if is_empty_value(v):
        return False
    text = str(v).strip()
    # Considera significativo si tiene al menos min_length caracteres y no es solo números/símbolos
    return len(text) >= min_length and bool(re.search(r'[a-záéíóúñ]', text.lower()))

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.dropna(how="all", axis=1)
    def row_empty(r):
        any_val = False
        for v in r:
            if is_meaningful_text(v, min_length=3):  # Menos restrictivo
                any_val = True
                break
        return not any_val
    mask = tmp.apply(row_empty, axis=1)
    return tmp.loc[~mask].reset_index(drop=True)

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Devuelve (objetivo_col, actividad_col) por heurística de nombre."""
    cols = list(df.columns)
    # candidatos de objetivo
    obj_candidates = [c for c in cols if any(k in c for k in [
        "objetivo especific", "objetivos especific", "objetivo espe", "objetivo 1", "obj 1", "especifico"
    ])]
    # candidatos de actividad
    act_candidates = [c for c in cols if any(k in c for k in [
        "actividad", "actividades objetivo", "actividad objetivo", "accion", "acciones"
    ])]
    obj_col = obj_candidates[0] if obj_candidates else (cols[0] if cols else None)
    # para actividad, preferir columnas distintas de obj
    act_candidates = [c for c in act_candidates if c != obj_col] or [c for c in cols if c != obj_col]
    act_col = act_candidates[0] if act_candidates else None
    return obj_col, act_col

def count_valid_pairs(df: pd.DataFrame, col_obj: str, col_act: str) -> int:
    """Cuenta pares válidos - solo requiere que la actividad tenga contenido"""
    return int((df[col_act].apply(lambda x: is_meaningful_text(x, min_length=5))).sum())

def count_all_activities(df: pd.DataFrame, col_act: str) -> int:
    """Cuenta todas las actividades con contenido significativo"""
    return int((df[col_act].apply(lambda x: is_meaningful_text(x, min_length=3))).sum())

# -------------------- Consistencia independiente (col2 vs col1) --------------------
def compute_pairwise_consistency_single(df: pd.DataFrame, name: str, col_obj: str, col_act: str, thresholds: Dict[str,float]):
    df = clean_rows(df.copy())
    scores, cats = [], []
    
    for _, row in df.iterrows():
        activity_text = str(row.get(col_act, ""))
        objective_text = str(row.get(col_obj, ""))
        
        # Si la actividad no tiene contenido significativo, skip
        if not is_meaningful_text(activity_text, min_length=3):
            continue
            
        # Si no hay objetivo, usar texto genérico para comparación
        if is_empty_value(objective_text):
            objective_text = "objetivo actividad universitaria educacion"
        
        a = normalize_text(objective_text)
        b = normalize_text(activity_text)
        
        s = fuzz.token_set_ratio(a, b)
        cat = classify_consistency(float(s), True, thresholds)
        scores.append(float(s))
        cats.append(cat)

    # Crear DataFrame solo con las filas procesadas
    valid_rows = []
    score_idx = 0
    
    for _, row in df.iterrows():
        if is_meaningful_text(str(row.get(col_act, "")), min_length=3):
            row_data = row.copy()
            row_data["col_objetivo"] = col_obj
            row_data["col_actividad"] = col_act
            row_data["score_obj_vs_act"] = scores[score_idx]
            row_data["clasificacion_calculada"] = cats[score_idx]
            valid_rows.append(row_data)
            score_idx += 1
    
    out = pd.DataFrame(valid_rows)
    
    counts = pd.Series(cats).value_counts(dropna=False).to_dict()
    total = len(valid_rows)
    summary = {
        "Fuente": name,
        "Total actividades": total,
        "Consistencia plena": int(counts.get("plena",0)),
        "Consistencia parcial": int(counts.get("parcial",0)),
        "Consistencia nula": int(counts.get("nula",0))
    }
    return summary, out

# -------------------- Consistencia contra PEI (PDF) --------------------
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
        for spec, sdata in (objdata.get("especificos", {}) or {}).items():
            acciones = sdata.get("acciones", []) or []
            indicadores = sdata.get("indicadores", []) or []
            if not acciones and not indicadores:
                index.append({"objetivo": obj, "especifico": spec, "texto": normalize_text(sdata.get("titulo","")), "tipo": "titulo"})
            else:
                for a in (acciones or [""]):
                    text = " ".join([a] + indicadores)
                    index.append({"objetivo": obj, "especifico": spec, "texto": normalize_text(text), "tipo": "accion_indicadores", "accion": a, "indicadores": indicadores})
    return [e for e in index if e.get("texto")]

def classify_consistency(score: float, ok_objective: bool, thresholds: Dict[str, float]) -> str:
    t_plena = thresholds.get("plena", 88.0)
    t_parcial = thresholds.get("parcial", 68.0)
    if score >= t_plena and ok_objective:
        return "plena"
    if score >= t_parcial:
        return "parcial"
    return "nula"

def compute_consistency_pei_single(df: pd.DataFrame, name: str, col_act: str, plan_index: List[Dict[str,Any]], thresholds: Dict[str,float]):
    df = clean_rows(df.copy())
    scores, cats, best_objs, best_specs = [], [], [], []
    
    valid_activities = []
    
    for _, row in df.iterrows():
        activity_text = str(row.get(col_act, ""))
        
        # Solo procesar actividades con contenido significativo
        if not is_meaningful_text(activity_text, min_length=3):
            continue
            
        valid_activities.append(row)
        text = normalize_text(activity_text)
        best, score = None, -1.0
        for entry in plan_index:
            s = fuzz.token_set_ratio(text, entry["texto"])
            if s > score:
                best, score = entry, s
        ok_objective = True
        cat = classify_consistency(float(score), ok_objective, thresholds)
        scores.append(float(score)); cats.append(cat)
        best_objs.append(best.get("objetivo") if best else None)
        best_specs.append(best.get("especifico") if best else None)
    
    out = pd.DataFrame(valid_activities).copy()
    out["pei_score"] = scores
    out["clasificacion_calculada"] = cats
    out["pei_objetivo"] = best_objs
    out["pei_especifico"] = best_specs

    counts = pd.Series(cats).value_counts(dropna=False).to_dict()
    total = len(valid_activities)
    summary = {
        "Fuente": name,
        "Total actividades": total,
        "Consistencia plena": int(counts.get("plena",0)),
        "Consistencia parcial": int(counts.get("parcial",0)),
        "Consistencia nula": int(counts.get("nula",0))
    }
    return summary, out

# -------------------- Reportes --------------------
def excel_from_blocks(blocks: List[Tuple[str, pd.DataFrame]]):
    from pandas import ExcelWriter
    bio = BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in blocks:
            df.to_excel(writer, index=False, sheet_name=sheet[:31])
    bio.seek(0)
    return bio.getvalue()
