from io import BytesIO
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from rapidfuzz import fuzz
from pypdf import PdfReader

# -------------------- Helpers --------------------
# Valores que indican "sin actividad propuesta para este objetivo"
NO_ACTIVITY_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "–", "—"}

SPANISH_STOP = set("""a al algo alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde dos el la los las en entre era erais eramos eran es esa esas ese eso esos esta estas este esto estos fue fuerais fuéramos fueran fui fuimos ha haber habia había habiais habíamos habían han has hasta hay la lo las los le les mas más me mi mis mucho muy nada ni no nos nosotras nosotros o os otra otras otro otros para pero poco por porque que quien quienes se sin sobre sois somos son soy su sus te tenia tenía teniais teníamos tenían tengo ti tu tus un una uno unas unos y ya""".split())

def strip_accents(s: str) -> str:
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')

def normalize_text(s: str) -> str:
    s = strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9áéíóúñ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in SPANISH_STOP and len(t) > 1]
    return " ".join(tokens)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        s = strip_accents(str(s)).lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out

def has_real_activity(v) -> bool:
    """Verifica si una celda contiene una actividad real (no 'None', etc.)"""
    if pd.isna(v):
        return False
    v_str = str(v).strip().lower()
    return v_str not in NO_ACTIVITY_TOKENS and len(v_str) > 0

def has_objective_assigned(v) -> bool:
    """Verifica si una celda tiene objetivo asignado"""
    if pd.isna(v):
        return False
    v_str = str(v).strip().lower()
    return v_str not in NO_ACTIVITY_TOKENS and len(v_str) > 0

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima - solo elimina filas completamente vacías"""
    if df is None or df.empty:
        return df
    return df.dropna(how='all').reset_index(drop=True)

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Devuelve (objetivo_col, actividad_col) por heurística de nombre."""
    cols = list(df.columns)
    obj_candidates = [c for c in cols if any(k in c for k in [
        "objetivo especific", "objetivos especific", "objetivo espe", "objetivo 1", "obj 1", "especifico"
    ])]
    act_candidates = [c for c in cols if any(k in c for k in [
        "actividad", "actividades objetivo", "actividad objetivo", "accion", "acciones"
    ])]
    obj_col = obj_candidates[0] if obj_candidates else (cols[0] if cols else None)
    act_candidates = [c for c in act_candidates if c != obj_col] or [c for c in cols if c != obj_col]
    act_col = act_candidates[0] if act_candidates else None
    return obj_col, act_col

def count_all_activity_cells(df: pd.DataFrame, col_act: str) -> Dict[str, int]:
    """Cuenta celdas de actividades por tipo"""
    total_cells = len(df)
    real_activities = df[col_act].apply(has_real_activity).sum()
    none_values = df[col_act].apply(lambda x: str(x).strip().lower() in NO_ACTIVITY_TOKENS).sum()
    empty_cells = total_cells - real_activities - none_values
    
    return {
        'total_cells': int(total_cells),
        'real_activities': int(real_activities),
        'none_values': int(none_values),
        'empty_cells': int(empty_cells)
    }

def count_activity_objective_pairs(df: pd.DataFrame, col_obj: str, col_act: str) -> Dict[str, int]:
    """Cuenta pares actividad-objetivo por tipo"""
    has_activity = df[col_act].apply(has_real_activity)
    has_objective = df[col_obj].apply(has_objective_assigned)
    
    return {
        'both_complete': int((has_activity & has_objective).sum()),
        'activity_only': int((has_activity & ~has_objective).sum()),
        'objective_only': int((~has_activity & has_objective).sum()),
        'both_empty': int((~has_activity & ~has_objective).sum())
    }

# -------------------- Consistencia por celdas individuales --------------------
def compute_pairwise_consistency_single(df: pd.DataFrame, name: str, col_obj: str, col_act: str, thresholds: Dict[str,float]):
    """Procesa cada celda individual de actividad"""
    df = clean_rows(df.copy())
    
    activities_data = []
    
    for idx, row in df.iterrows():
        activity_text = str(row.get(col_act, ""))
        objective_text = str(row.get(col_obj, ""))
        
        # Solo procesar celdas que contienen actividades reales
        if not has_real_activity(activity_text):
            continue
            
        # Para actividades reales sin objetivo, usar genérico
        if not has_objective_assigned(objective_text):
            objective_text = "objetivo actividad universitaria educacion plan estrategico institucional"
        
        # Normalizar textos
        a = normalize_text(objective_text)
        b = normalize_text(activity_text)
        
        # Valores por defecto si la normalización elimina todo
        if not a.strip():
            a = "objetivo universitario educacion plan"
        if not b.strip():
            b = "actividad educativa tarea universitaria"
            
        score = fuzz.token_set_ratio(a, b)
        classification = classify_consistency(float(score), True, thresholds)
        
        activities_data.append({
            col_obj: objective_text,
            col_act: activity_text,
            'col_objetivo': col_obj,
            'col_actividad': col_act,
            'score_obj_vs_act': float(score),
            'clasificacion_calculada': classification,
            'fila_original': idx
        })
    
    # Crear DataFrame con actividades individuales
    out = pd.DataFrame(activities_data)
    
    if len(activities_data) == 0:
        summary = {
            "Fuente": name,
            "Total actividades": 0,
            "Consistencia plena": 0,
            "Consistencia parcial": 0,
            "Consistencia nula": 0
        }
        return summary, out
    
    # Contar por clasificación
    counts = out['clasificacion_calculada'].value_counts().to_dict()
    total = len(activities_data)
    
    summary = {
        "Fuente": name,
        "Total actividades": total,
        "Consistencia plena": int(counts.get("plena", 0)),
        "Consistencia parcial": int(counts.get("parcial", 0)),
        "Consistencia nula": int(counts.get("nula", 0))
    }
    
    return summary, out

# -------------------- Consistencia contra PEI --------------------
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
    """Analiza cada actividad individual contra el PEI"""
    df = clean_rows(df.copy())
    
    activities_data = []
    
    for idx, row in df.iterrows():
        activity_text = str(row.get(col_act, ""))
        
        # Solo procesar actividades reales
        if not has_real_activity(activity_text):
            continue
            
        text = normalize_text(activity_text)
        if not text.strip():
            text = "actividad educativa universitaria plan"
            
        best, score = None, -1.0
        for entry in plan_index:
            s = fuzz.token_set_ratio(text, entry["texto"])
            if s > score:
                best, score = entry, s
                
        classification = classify_consistency(float(score), True, thresholds)
        
        activities_data.append({
            col_act: activity_text,
            'pei_score': float(score),
            'clasificacion_calculada': classification,
            'pei_objetivo': best.get("objetivo") if best else None,
            'pei_especifico': best.get("especifico") if best else None,
            'fila_original': idx
        })
    
    out = pd.DataFrame(activities_data)
    
    if len(activities_data) == 0:
        summary = {
            "Fuente": name,
            "Total actividades": 0,
            "Consistencia plena": 0,
            "Consistencia parcial": 0,
            "Consistencia nula": 0
        }
        return summary, out
    
    counts = out['clasificacion_calculada'].value_counts().to_dict()
    total = len(activities_data)
    
    summary = {
        "Fuente": name,
        "Total actividades": total,
        "Consistencia plena": int(counts.get("plena", 0)),
        "Consistencia parcial": int(counts.get("parcial", 0)),
        "Consistencia nula": int(counts.get("nula", 0))
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
