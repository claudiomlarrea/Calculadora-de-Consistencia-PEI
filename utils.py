from io import BytesIO
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from rapidfuzz import fuzz
from pypdf import PdfReader

# -------------------- Helpers --------------------
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
    """Limpieza mínima"""
    if df is None or df.empty:
        return df
    return df.dropna(how='all').reset_index(drop=True)

def detect_all_objective_activity_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Detecta TODOS los pares de columnas (objetivo, actividad) en el formulario"""
    cols = list(df.columns)
    pairs = []
    
    # Buscar patrones de objetivos específicos
    obj_patterns = [
        r"objetivos?\s+especific\w*\s*(\d+)",
        r"objetivo\s+(\d+)",
        r"obj\s+(\d+)",
        r"especific\w*\s+(\d+)"
    ]
    
    # Buscar patrones de actividades
    act_patterns = [
        r"actividad\w*\s+objetivo\s*(\d+)",
        r"actividad\w*\s+(\d+)",
        r"accion\w*\s+(\d+)"
    ]
    
    # Encontrar columnas de objetivos numeradas
    obj_cols = {}
    for col in cols:
        for pattern in obj_patterns:
            match = re.search(pattern, col, re.IGNORECASE)
            if match:
                num = match.group(1)
                obj_cols[num] = col
                break
    
    # Encontrar columnas de actividades numeradas
    act_cols = {}
    for col in cols:
        for pattern in act_patterns:
            match = re.search(pattern, col, re.IGNORECASE)
            if match:
                num = match.group(1)
                act_cols[num] = col
                break
    
    # Emparejar por número
    for num in obj_cols:
        if num in act_cols:
            pairs.append((obj_cols[num], act_cols[num]))
    
    # Si no encontramos pares numerados, usar detección simple
    if not pairs:
        pairs = [detect_columns(df)]
    
    return [p for p in pairs if p[0] and p[1]]

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detección simple de una columna objetivo y una actividad"""
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

def analyze_participant_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza completitud por participante considerando TODOS los pares objetivo-actividad"""
    pairs = detect_all_objective_activity_pairs(df)
    
    if not pairs:
        return {
            'total_participants': len(df),
            'complete_participants': 0,
            'incomplete_participants': len(df),
            'total_activities': 0,
            'pairs_detected': 0
        }
    
    complete_participants = 0
    total_activities = 0
    
    for _, row in df.iterrows():
        participant_has_activity = False
        
        # Revisar todos los pares objetivo-actividad para este participante
        for obj_col, act_col in pairs:
            obj_val = row.get(obj_col, "")
            act_val = row.get(act_col, "")
            
            # Si tiene tanto objetivo como actividad válidos
            if has_objective_assigned(obj_val) and has_real_activity(act_val):
                participant_has_activity = True
                total_activities += 1
        
        if participant_has_activity:
            complete_participants += 1
    
    return {
        'total_participants': len(df),
        'complete_participants': complete_participants,
        'incomplete_participants': len(df) - complete_participants,
        'total_activities': total_activities,
        'pairs_detected': len(pairs)
    }

def extract_all_activities(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extrae TODAS las actividades de todos los pares objetivo-actividad"""
    pairs = detect_all_objective_activity_pairs(df)
    activities = []
    
    for participant_idx, row in df.iterrows():
        for obj_col, act_col in pairs:
            obj_val = row.get(obj_col, "")
            act_val = row.get(act_col, "")
            
            # Solo incluir si tiene actividad real
            if has_real_activity(act_val):
                activities.append({
                    'participant_id': participant_idx,
                    'objetivo_col': obj_col,
                    'actividad_col': act_col,
                    'objetivo_text': str(obj_val),
                    'actividad_text': str(act_val),
                    'has_objetivo': has_objective_assigned(obj_val)
                })
    
    return activities

# -------------------- Consistencia --------------------
def compute_pairwise_consistency_single(df: pd.DataFrame, name: str, col_obj: str, col_act: str, thresholds: Dict[str,float]):
    """Procesa todas las actividades de todos los pares objetivo-actividad"""
    df = clean_rows(df.copy())
    
    # En lugar de usar solo las columnas seleccionadas, extraer todas las actividades
    all_activities = extract_all_activities(df)
    
    if not all_activities:
        summary = {
            "Fuente": name,
            "Total actividades": 0,
            "Consistencia plena": 0,
            "Consistencia parcial": 0,
            "Consistencia nula": 0
        }
        return summary, pd.DataFrame()
    
    activities_data = []
    
    for activity in all_activities:
        objective_text = activity['objetivo_text']
        activity_text = activity['actividad_text']
        
        # Si no tiene objetivo, usar genérico
        if not activity['has_objetivo']:
            objective_text = "objetivo actividad universitaria educacion plan estrategico institucional"
        
        # Normalizar textos
        a = normalize_text(objective_text)
        b = normalize_text(activity_text)
        
        # Valores por defecto
        if not a.strip():
            a = "objetivo universitario educacion plan"
        if not b.strip():
            b = "actividad educativa tarea universitaria"
            
        score = fuzz.token_set_ratio(a, b)
        classification = classify_consistency(float(score), True, thresholds)
        
        activities_data.append({
            'objetivo_original': activity['objetivo_col'],
            'actividad_original': activity['actividad_col'],
            'objetivo_text': objective_text,
            'actividad_text': activity_text,
            'score_obj_vs_act': float(score),
            'clasificacion_calculada': classification,
            'participant_id': activity['participant_id'],
            'has_objetivo': activity['has_objetivo']
        })
    
    out = pd.DataFrame(activities_data)
    
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

# -------------------- PEI --------------------
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
    """Analiza todas las actividades extraídas contra el PEI"""
    df = clean_rows(df.copy())
    
    all_activities = extract_all_activities(df)
    
    if not all_activities:
        summary = {
            "Fuente": name,
            "Total actividades": 0,
            "Consistencia plena": 0,
            "Consistencia parcial": 0,
            "Consistencia nula": 0
        }
        return summary, pd.DataFrame()
    
    activities_data = []
    
    for activity in all_activities:
        activity_text = activity['actividad_text']
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
            'actividad_text': activity_text,
            'pei_score': float(score),
            'clasificacion_calculada': classification,
            'pei_objetivo': best.get("objetivo") if best else None,
            'pei_especifico': best.get("especifico") if best else None,
            'participant_id': activity['participant_id']
        })
    
    out = pd.DataFrame(activities_data)
    
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
