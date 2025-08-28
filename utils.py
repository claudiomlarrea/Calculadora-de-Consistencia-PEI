
from io import BytesIO
import re
import unicodedata
from typing import Dict, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# PDF reader opcional (no rompe si falta)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

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
       - resumen (dict) con totales y consistencia
       - detalle (DataFrame) con columnas extra pedidas
       - cons_obj (DataFrame consistencia por objetivo)
       - mejoras (DataFrame sugerencias con Δ p.p. >= 15)
       - duplicadas (DataFrame sospecha de duplicados)
    """
    df = clean_rows(df.copy())
    total_filas = len(df)

    valid_mask = (~df[col_obj].apply(is_empty_value)) & (~df[col_act].apply(is_empty_value))
    data = df.loc[valid_mask].reset_index(drop=True).copy()
    con_detalle = len(data)
    sin_detalle = int(total_filas - con_detalle)

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
    data["% actual (objetivo↔actividad)"] = [round(x,1) for x in scores]
    data["clasificacion"] = cats
    data["Δ p.p. (objetivo propuesto - actual)"] = [round(x,1) for x in deltas]

    # Resumen
    counts = data["clasificacion"].value_counts(dropna=False).to_dict()
    total_validas = len(data)
    resumen = {
        "Fuente": name,
        "Total participantes": int(total_filas),
        "Con detalle": int(con_detalle),
        "Sin detalle": int(sin_detalle),
        "Total actividades (evaluadas)": int(total_validas),
        "Consistencia plena": int(counts.get("plena",0)),
        "Consistencia parcial": int(counts.get("parcial",0)),
        "Consistencia nula": int(counts.get("nula",0))
    }

    # Consistencia por objetivo
    g = data.groupby(col_obj)["% actual (objetivo↔actividad)"].agg(["count","mean","median"])
    g["p25"] = data.groupby(col_obj)["% actual (objetivo↔actividad)"].quantile(0.25)
    g["p75"] = data.groupby(col_obj)["% actual (objetivo↔actividad)"].quantile(0.75)
    g["std"] = data.groupby(col_obj)["% actual (objetivo↔actividad)"].std(ddof=0)
    low = data.assign(_low = data["clasificacion"].eq("nula")).groupby(col_obj)["_low"].mean().rename("% en Bajo")
    cons_obj = g.join(low).reset_index().rename(columns={col_obj:"Objetivo específico","count":"N","mean":"Promedio","median":"Mediana"})
    cons_obj["% en Bajo"] = (cons_obj["% en Bajo"]*100).round(1)
    cons_obj = cons_obj.sort_values("% en Bajo", ascending=False)

    # Potencial de reubicación (ganancia >= 15 p.p.)
    mejoras = data.copy()
    mejoras = mejoras.loc[mejoras["Δ p.p. (objetivo propuesto - actual)"] >= 15.0].sort_values("Δ p.p. (objetivo propuesto - actual)", ascending=False)
    mejoras = mejoras[[
        "Actividad (seleccionada)",  # actividad
        col_obj,                      # objetivo actual
        "% actual (objetivo↔actividad)",  # % actual
        "Mejor objetivo propuesto",   # objetivo sugerido
        "% si se ubica en objetivo propuesto",  # % sugerido
        "Δ p.p. (objetivo propuesto - actual)"
    ]].rename(columns={
        "Actividad (seleccionada)":"Actividad",
        col_obj:"Obj. actual",
        "% actual (objetivo↔actividad)":"% actual",
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

    return resumen, data, cons_obj, mejoras, dup

# -------------------- Exportadores --------------------
def excel_consolidado(resumen: Dict[str,int], detalle: pd.DataFrame,
                      cons_obj: pd.DataFrame, mejoras: pd.DataFrame, duplicadas: pd.DataFrame):
    """Excel con Consistencia (no 'Rendimiento')."""
    from pandas import ExcelWriter
    bio = BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        # Resumen
        pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
        # Porcentajes
        T = max(resumen.get("Total actividades (evaluadas)", 0), 1)
        p = pd.DataFrame([{
            "Fuente": resumen.get("Fuente",""),
            "% Consistencia plena": round(resumen.get("Consistencia plena",0)/T,4),
            "% Consistencia parcial": round(resumen.get("Consistencia parcial",0)/T,4),
            "% Consistencia nula": round(resumen.get("Consistencia nula",0)/T,4),
        }])
        p.to_excel(writer, index=False, sheet_name="Porcentajes")
        # Consistencia por objetivo (nuevo nombre)
        cons_obj.to_excel(writer, index=False, sheet_name="Consistencia_por_objetivo")
        mejoras.to_excel(writer, index=False, sheet_name="Potencial_reubicacion")
        duplicadas.to_excel(writer, index=False, sheet_name="Duplicadas")
        detalle.to_excel(writer, index=False, sheet_name="Detalle")
    bio.seek(0)
    return bio.getvalue()

def docx_conclusiones(resumen: Dict[str,int], detalle: pd.DataFrame, cons_obj: pd.DataFrame, mejoras: pd.DataFrame):
    """Diagnóstico Completo del Formulario (Word)."""
    try:
        from docx import Document
        from docx.shared import Pt
    except Exception:
        return None

    def pct(a,b): return 0 if b==0 else round(100*a/b,1)

    # Métricas globales
    scores = detalle["% actual (objetivo↔actividad)"] if not detalle.empty else pd.Series(dtype=float)
    mean = round(float(scores.mean()) if len(scores) else 0,1)
    med  = round(float(scores.median()) if len(scores) else 0,1)
    p25  = round(float(scores.quantile(0.25)) if len(scores) else 0,1)
    p75  = round(float(scores.quantile(0.75)) if len(scores) else 0,1)
    rng_min = round(float(scores.min()) if len(scores) else 0,1)
    rng_max = round(float(scores.max()) if len(scores) else 0,1)

    T_part = int(resumen.get("Total participantes",0))
    con_det = int(resumen.get("Con detalle",0))
    sin_det = int(resumen.get("Sin detalle",0))
    T_eval = int(resumen.get("Total actividades (evaluadas)",0))
    P = int(resumen.get("Consistencia plena",0))
    Pa = int(resumen.get("Consistencia parcial",0))
    N = int(resumen.get("Consistencia nula",0))

    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Times New Roman"; style.font.size = Pt(11)
    doc.add_heading("Diagnóstico Completo del Formulario", level=1)

    # Totales
    doc.add_paragraph(f"Total Participantes {T_part}")
    doc.add_paragraph(f"Con Detalle {con_det}")
    doc.add_paragraph(f"Sin Detalle {sin_det}")
    doc.add_paragraph(f"Total Actividades completas {T_eval}")

    doc.add_heading("Resumen de Participación", level=2)
    doc.add_paragraph(f"Total participantes: {T_part} actividades")
    doc.add_paragraph(f"Actividades con detalle: {con_det} actividades ({pct(con_det, T_part)}%)")
    doc.add_paragraph(f"Sin detalle: {sin_det} actividades")

    doc.add_heading("Estadísticas de Actividades", level=2)
    doc.add_paragraph(f"Total actividades completas propuestas: {T_eval}")
    doc.add_paragraph(f"Promedio de consistencia: {mean}%")
    doc.add_paragraph(f"Mediana: {med}% | P25: {p25}% | P75: {p75}%")
    doc.add_paragraph(f"Rango: {rng_min}% - {rng_max}%")

    doc.add_heading("Distribución por Niveles de Consistencia", level=2)
    table = doc.add_table(rows=1, cols=3)
    hdr = table.rows[0].cells
    hdr[0].text = "Nivel"; hdr[1].text = "N actividades"; hdr[2].text = "Porcentaje"
    rows = [("Alta (≥75%)", P, pct(P,T_eval)), ("Media (50–74%)", Pa, pct(Pa,T_eval)), ("Baja (<50%)", N, pct(N,T_eval))]
    for r in rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = str(r[0]), str(r[1]), f"{r[2]}%"

    # Top 10 objetivos más críticos por % en Bajo
    if not cons_obj.empty:
        doc.add_heading("Consistencia por Objetivo específico (Top 10 críticos)", level=2)
        top = cons_obj.sort_values("% en Bajo", ascending=False).head(10)
        t = doc.add_table(rows=1, cols=5)
        t.rows[0].cells[0].text = "Objetivo específico"
        t.rows[0].cells[1].text = "N"
        t.rows[0].cells[2].text = "Promedio"
        t.rows[0].cells[3].text = "Mediana"
        t.rows[0].cells[4].text = "% en Bajo"
        for _, r in top.iterrows():
            cells = t.add_row().cells
            cells[0].text = str(r["Objetivo específico"]); cells[1].text = str(int(r["N"]))
            cells[2].text = f"{round(float(r['Promedio']),1)}%"; cells[3].text = f"{round(float(r['Mediana']),1)}%"; cells[4].text = f"{round(float(r['% en Bajo']),1)}%"

    # Top 15 sugerencias
    if not mejoras.empty:
        doc.add_heading("Actividades con alto potencial de mejora/reubicación", level=2)
        k = min(15, len(mejoras))
        t = doc.add_table(rows=1, cols=6)
        t.rows[0].cells[0].text = "Actividad"
        t.rows[0].cells[1].text = "Obj. actual"
        t.rows[0].cells[2].text = "% actual"
        t.rows[0].cells[3].text = "Obj. sugerido"
        t.rows[0].cells[4].text = "% sugerido"
        t.rows[0].cells[5].text = "Δ p.p."
        for _, r in mejoras.head(k).iterrows():
            cells = t.add_row().cells
            cells[0].text = str(r["Actividad"])
            cells[1].text = str(r["Obj. actual"])
            cells[2].text = f"{round(float(r['% actual']),1)}%"
            cells[3].text = str(r["Obj. sugerido"])
            cells[4].text = f"{round(float(r['% sugerido']),1)}%"
            cells[5].text = f"{round(float(r['Δ p.p.']),1)}"

    bio = BytesIO(); doc.save(bio); bio.seek(0)
    return bio.getvalue()
