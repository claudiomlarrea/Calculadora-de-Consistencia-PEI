
import io
import re
import unicodedata
import datetime as dt
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# ------------- Similaridad robusta (rapidfuzz si está, difflib si no) -------------
try:
    from rapidfuzz import fuzz as _rf_fuzz
    def token_set_ratio(a: str, b: str) -> float:
        return float(_rf_fuzz.token_set_ratio(a, b))
except Exception:
    from difflib import SequenceMatcher
    def token_set_ratio(a: str, b: str) -> float:
        sa = " ".join(sorted(set(str(a).split())))
        sb = " ".join(sorted(set(str(b).split())))
        return 100.0 * SequenceMatcher(None, sa, sb).ratio()

# -------------------- Normalización y utilidades --------------------
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
        return all(is_empty_value(v) for v in r)
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

def _obj_general_token(obj_text: str) -> str:
    s = str(obj_text).strip()
    m = re.match(r"^\s*([1-9])(\.|[\s])", s)
    return f"Obj {m.group(1)}" if m else "OTROS/SIN ID"

def classify(score: float, thr_full=88.0, thr_partial=68.0) -> str:
    if score >= thr_full: return "plena"
    if score >= thr_partial: return "parcial"
    return "nula"

def best_alt_objective(activity_norm: str, current_obj: str, unique_objs_norm):
    best_txt, best_score = None, -1.0
    cur_norm = normalize_text(current_obj)
    for raw, norm in unique_objs_norm:
        if norm == cur_norm: continue
        s = token_set_ratio(activity_norm, norm)
        if s > best_score:
            best_score = s; best_txt = raw
    return best_txt, float(best_score)

def analyze_independent(df: pd.DataFrame, name: str, col_obj: str, col_act: str, thr_full=88.0, thr_partial=68.0):
    df = clean_rows(df.copy())
    total_cargadas = len(df)

    df["_obj_txt"] = df[col_obj].astype(str)
    df["_act_txt"] = df[col_act].astype(str)
    df["_obj_norm"] = df["_obj_txt"].map(normalize_text)
    df["_act_norm"] = df["_act_txt"].map(normalize_text)

    valid_mask = (~df[col_obj].apply(is_empty_value)) & (~df[col_act].apply(is_empty_value))
    valid = df.loc[valid_mask].copy().reset_index(drop=True)

    unique_objs = list(dict.fromkeys(valid[col_obj].astype(str).tolist()))
    unique_objs_norm = [(t, normalize_text(t)) for t in unique_objs]

    score_arr = np.full(total_cargadas, np.nan, dtype=float)
    cat_arr   = np.array(["sin_datos"]*total_cargadas, dtype=object)
    alt_obj   = np.array([None]*total_cargadas, dtype=object)
    alt_score = np.full(total_cargadas, np.nan, dtype=float)
    delta_pp  = np.full(total_cargadas, np.nan, dtype=float)

    for idx_valid, row in valid.iterrows():
        idx = valid_mask[valid_mask].index[idx_valid]
        s = token_set_ratio(row["_obj_norm"], row["_act_norm"])
        cat = classify(float(s), thr_full, thr_partial)
        bo, bs = best_alt_objective(row["_act_norm"], row[col_obj], unique_objs_norm)
        score_arr[idx] = float(s)
        cat_arr[idx]   = cat
        alt_obj[idx]   = bo
        alt_score[idx] = float(bs)
        delta_pp[idx]  = float(bs - s)

    detalle_full = df[[col_obj, col_act]].copy()
    detalle_full.rename(columns={col_obj:"Objetivo específico", col_act:"Actividad (seleccionada)"}, inplace=True)
    detalle_full["% actual (objetivo↔actividad)"] = np.round(score_arr, 1)
    detalle_full["clasificacion"] = cat_arr
    detalle_full["Mejor objetivo propuesto"] = alt_obj
    detalle_full["% si se ubica en objetivo propuesto"] = np.round(alt_score, 1)
    detalle_full["Δ p.p. (objetivo propuesto - actual)"] = np.round(delta_pp, 1)
    detalle_full["Objetivo general (1..n)"] = detalle_full["Objetivo específico"].map(_obj_general_token)

    counts = pd.Series(cat_arr).value_counts(dropna=False).to_dict()
    total_validas = int(valid_mask.sum())
    resumen = {
        "Fuente": name,
        "Total actividades (cargadas)": int(total_cargadas),
        "Total actividades (evaluadas)": int(total_validas),
        "Consistencia plena": int(counts.get("plena",0)),
        "Consistencia parcial": int(counts.get("parcial",0)),
        "Consistencia nula": int(counts.get("nula",0)),
        "Sin datos": int(counts.get("sin_datos",0)),
    }

    cons_obj = (detalle_full.loc[detalle_full["clasificacion"]!="sin_datos"]
                .groupby("Objetivo específico")["% actual (objetivo↔actividad)"]
                .agg(["count","mean","median"])
                .rename(columns={"count":"N","mean":"Promedio","median":"Mediana"}))
    if not cons_obj.empty:
        cons_obj["p25"] = (detalle_full.loc[detalle_full["clasificacion"]!="sin_datos"]
                           .groupby("Objetivo específico")["% actual (objetivo↔actividad)"].quantile(0.25))
        cons_obj["p75"] = (detalle_full.loc[detalle_full["clasificacion"]!="sin_datos"]
                           .groupby("Objetivo específico")["% actual (objetivo↔actividad)"].quantile(0.75))
        cons_obj["std"] = (detalle_full.loc[detalle_full["clasificacion"]!="sin_datos"]
                           .groupby("Objetivo específico")["% actual (objetivo↔actividad)"].std(ddof=0))
        low = (detalle_full.assign(_low = detalle_full["clasificacion"].eq("nula"))
               .loc[detalle_full["clasificacion"]!="sin_datos"]
               .groupby("Objetivo específico")["_low"].mean().rename("% en Bajo"))
        cons_obj = cons_obj.join(low)
        cons_obj["% en Bajo"] = (cons_obj["% en Bajo"]*100).round(1)
        cons_obj = cons_obj.reset_index().sort_values("% en Bajo", ascending=False)
    else:
        cons_obj = pd.DataFrame(columns=["Objetivo específico","N","Promedio","Mediana","p25","p75","std","% en Bajo"])

    mejoras = detalle_full.loc[detalle_full["clasificacion"]!="sin_datos"].copy()
    mejoras = mejoras.loc[mejoras["Δ p.p. (objetivo propuesto - actual)"] >= 15.0]
    mejoras = mejoras.sort_values("Δ p.p. (objetivo propuesto - actual)", ascending=False)
    mejoras = mejoras[[
        "Actividad (seleccionada)",
        "Objetivo específico",
        "% actual (objetivo↔actividad)",
        "Mejor objetivo propuesto",
        "% si se ubica en objetivo propuesto",
        "Δ p.p. (objetivo propuesto - actual)"
    ]].rename(columns={
        "Actividad (seleccionada)":"Actividad",
        "Objetivo específico":"Obj. actual",
        "% actual (objetivo↔actividad)":"% actual",
        "Mejor objetivo propuesto":"Obj. sugerido",
        "% si se ubica en objetivo propuesto":"% sugerido",
        "Δ p.p. (objetivo propuesto - actual)":"Δ p.p."
    })
    mejoras["% actual"] = mejoras["% actual"].round(1)
    mejoras["% sugerido"] = mejoras["% sugerido"].round(1)
    mejoras["Δ p.p."] = mejoras["Δ p.p."].round(1)

    dup = (df.assign(act_norm=df["_act_norm"])
             .groupby("act_norm").size().reset_index(name="Repeticiones"))
    dup = dup.loc[dup["Repeticiones"]>1].sort_values("Repeticiones", ascending=False)
    dup = dup.rename(columns={"act_norm":"Actividad (normalizada)"})

    control = (detalle_full.groupby("Objetivo general (1..n)").size()
               .reset_index(name="Total filas")).sort_values("Objetivo general (1..n)")
    control_total = pd.DataFrame([{"Objetivo general (1..n)":"TOTAL", "Total filas": int(total_cargadas)}])
    control = pd.concat([control, control_total], ignore_index=True)

    return resumen, detalle_full, cons_obj, mejoras, dup, control

def excel_consolidado(resumen: Dict[str,int], detalle_full: pd.DataFrame,
                      cons_obj: pd.DataFrame, mejoras: pd.DataFrame, duplicadas: pd.DataFrame, control: pd.DataFrame):
    from pandas import ExcelWriter
    bio = io.BytesIO()
    with ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
        T = max(resumen.get("Total actividades (evaluadas)", 0), 1)
        p = pd.DataFrame([{
            "Fuente": resumen.get("Fuente",""),
            "% Consistencia plena": round(resumen.get("Consistencia plena",0)/T,4),
            "% Consistencia parcial": round(resumen.get("Consistencia parcial",0)/T,4),
            "% Consistencia nula": round(resumen.get("Consistencia nula",0)/T,4),
            "% Sin datos (sobre cargadas)": round(resumen.get("Sin datos",0)/max(resumen.get("Total actividades (cargadas)",1),1),4),
        }])
        p.to_excel(writer, index=False, sheet_name="Porcentajes")
        cons_obj.to_excel(writer, index=False, sheet_name="Consistencia_por_objetivo")
        mejoras.to_excel(writer, index=False, sheet_name="Potencial_reubicacion")
        duplicadas.to_excel(writer, index=False, sheet_name="Duplicadas")
        detalle_full.to_excel(writer, index=False, sheet_name="Actividades")
        control.to_excel(writer, index=False, sheet_name="Control_conteo")
    bio.seek(0)
    return bio.getvalue()

# ---------- UI Streamlit (single-file) ----------
st.set_page_config(page_title="Calculadora – Formulario Único", layout="wide")
st.title("📊 Calculadora de Consistencia – Formulario Único (1 archivo)")

uploaded = st.file_uploader("Subí el **Formulario Único** (XLSX o CSV)", type=["xlsx","csv"])
if not uploaded:
    st.stop()

bio = io.BytesIO(uploaded.getvalue()); bio.name = uploaded.name
if uploaded.name.lower().endswith(".xlsx"):
    df = pd.read_excel(bio, engine="openpyxl")
else:
    try:
        df = pd.read_csv(bio, encoding="utf-8", sep=None, engine="python")
    except Exception:
        bio.seek(0); df = pd.read_csv(bio, encoding="latin-1", sep=None, engine="python")

df = normalize_colnames(df)
df = clean_rows(df)

obj_default, act_default = detect_columns(df)
st.subheader("Seleccioná columnas")
c1, c2 = st.columns(2)
with c1:
    col_obj = st.selectbox("Columna de **Objetivo específico**", options=list(df.columns),
                           index=(list(df.columns).index(obj_default) if obj_default in df.columns else 0))
with c2:
    col_act = st.selectbox("Columna de **Actividad**", options=list(df.columns),
                           index=(list(df.columns).index(act_default) if act_default in df.columns else (1 if len(df.columns)>1 else 0)))

total_valid = count_valid_pairs(df, col_obj, col_act)
st.info(f"Total de actividades **cargadas**: {len(df)} | **evaluadas** (objetivo + actividad): {total_valid}")

st.subheader("Vista previa")
st.dataframe(df[[col_obj, col_act]].head(15), use_container_width=True)

st.subheader("Umbrales de clasificación")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena'", 70, 100, 88, 1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", 50, 90, 68, 1)

if st.button("🔎 Realizar Análisis Completo de Consistencia"):
    with st.spinner("Calculando consistencias y generando informe Excel…"):
        resumen, detalle_full, cons_obj, mejoras, dup, control = analyze_independent(
            df, uploaded.name, col_obj, col_act, thr_full=float(t_plena), thr_partial=float(t_parcial)
        )

    st.subheader("📊 Resumen")
    st.write(resumen)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_bytes = excel_consolidado(resumen, detalle_full, cons_obj, mejoras, dup, control)

    st.download_button("⬇️ Descargar EXCEL — informe_consistencia_pei_consolidado",
                       data=excel_bytes,
                       file_name=f"informe_consistencia_pei_consolidado_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.warning("Cargá el archivo y tocá el botón para generar los informes.")
