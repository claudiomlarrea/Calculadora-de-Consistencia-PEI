
import io
import re
import unicodedata
import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

SPANISH_STOPWORDS = set("""
a al algo alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde dos el 
ella ellas ello ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estais estaban estabas 
estad estada estado estais estamos estan estar estara estas este esto estos estoy fue fueron fui fuimos ha habiendo 
habla hablar hablaron hace hacen hacer hacerlo hacia han hasta hay haya he la las le les lo los mas me mi mia mias 
mientras mio mios mis mucha muchas mucho muchos muy nada ni no nos nosotras nosotros nuestra nuestras nuestro nuestros 
nunca o os otra otras otro otros para pero poco por porque podria puedo pues que quien quienes se sea sean segun ser 
si siempre sin sobre sr sra sres su sus te tener tiene tienen toda todas todo todos tu tus un una uno unos usted ustedes 
ya y cuáles cuál cuál/cuál qué que/qué quién quiénes dónde cuándo cómo porqué también sólo según 
""".split())

def normalize_text(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str):
    text = normalize_text(text)
    toks = [t for t in text.split() if t not in SPANISH_STOPWORDS and len(t) > 2]
    return toks

def keyword_overlap_score(activity_text, objective_text):
    a_tokens = set(tokenize(activity_text))
    o_tokens = set(tokenize(objective_text))
    if not a_tokens or not o_tokens:
        return 0.0
    overlap = len(a_tokens & o_tokens)
    denom = np.sqrt(len(a_tokens) * len(o_tokens))
    return overlap / denom

def top_overlap_terms(activity_text, objective_text, topn=6):
    a_tokens = tokenize(activity_text)
    o_tokens = set(tokenize(objective_text))
    common = [t for t in a_tokens if t in o_tokens]
    from collections import Counter
    freq = Counter(common)
    return ", ".join([w for w, _ in freq.most_common(topn)])

# ------------- Formulario Único -> formato largo -------------
def maybe_from_formulario_unico(df_raw: pd.DataFrame):
    """Detecta estructura 'Formulario Único' (ancha) y la transforma en
    un DataFrame largo con columnas estándar: 'Actividad', 'Obj. actual', 'Obj. sugerido'.
    Si no coincide el patrón, devuelve df_raw sin cambios.
    """
    cols = df_raw.columns.astype(str)
    has_blocks = any(re.search(r'Objetivos?\s*espec[ií]ficos\s*\d+', c, re.I) for c in cols) and                      any(re.search(r'Actividades\s*Objetivo\s*\d+', c, re.I) for c in cols)
    if not has_blocks:
        return df_raw  # no es el formulario único (o no coinciden encabezados)

    # recolectar tripletas (i, obj, act, det)
    blocks = []
    for i in range(1, 21):
        col_obj = next((c for c in cols if re.search(fr'Objetivos?\s*espec[ií]ficos\s*{i}\b', c, re.I)), None)
        col_act = next((c for c in cols if re.search(fr'Actividades\s*Objetivo\s*{i}\b', c, re.I)), None)
        col_det = next((c for c in cols if re.search(fr'Detalle.*Objetivo\s*{i}\b', c, re.I)), None)
        if col_obj or col_act or col_det:
            blocks.append((i, col_obj, col_act, col_det))
    if not blocks:
        return df_raw

    tidy_rows = []
    meta_cols = [c for c in cols if re.search(r'^(AÑO|.*Unidad Acad[eé]mica|Direcci[oó]n de correo|Marca temporal|Puntuaci[oó]n)$', c, re.I)]

    for _, row in df_raw.iterrows():
        meta = {c: row.get(c, None) for c in meta_cols}
        for i, col_obj, col_act, col_det in blocks:
            obj = row.get(col_obj, None) if col_obj in df_raw.columns else None
            act = row.get(col_act, None) if col_act in df_raw.columns else None
            det = row.get(col_det, None) if col_det in df_raw.columns else None
            if (pd.notna(obj) and str(obj).strip()) or (pd.notna(act) and str(act).strip()) or (pd.notna(det) and str(det).strip()):
                actividad_txt = ""
                if pd.notna(act) and str(act).strip():
                    actividad_txt += str(act).strip()
                if pd.notna(det) and str(det).strip():
                    actividad_txt += (" — " if actividad_txt else "") + str(det).strip()
                d = {
                    "Actividad": actividad_txt if actividad_txt else None,
                    "Obj. actual": str(obj).strip() if pd.notna(obj) else None,
                    "Obj. sugerido": ""
                }
                d.update(meta)
                tidy_rows.append(d)
    tidy_df = pd.DataFrame(tidy_rows)
    # limpieza mínima
    if "Actividad" in tidy_df.columns:
        tidy_df["Actividad"] = tidy_df["Actividad"].fillna("").astype(str)
    if "Obj. actual" in tidy_df.columns:
        tidy_df["Obj. actual"] = tidy_df["Obj. actual"].fillna("").astype(str)
    return tidy_df

# ------------- Detección de columnas estándar -------------
def detect_columns(df: pd.DataFrame):
    col_act = None
    for c in df.columns:
        if re.search(r'(\bactividad\b|descripci[oó]n)', str(c), re.I):
            col_act = c; break
    col_obj_actual = None
    for c in df.columns:
        if re.search(r'(obj.*actual)', str(c), re.I):
            col_obj_actual = c; break
    col_obj_sug = None
    for c in df.columns:
        if re.search(r'(obj.*sugerid)', str(c), re.I):
            col_obj_sug = c; break
    return col_act, col_obj_actual, col_obj_sug

def build_objective_catalog(df, col_obj_actual, col_obj_sug, cat_df=None):
    if cat_df is not None and not cat_df.empty:
        name_col = None; desc_col = None
        for c in cat_df.columns:
            if re.search(r'(objetivo|nombre|t[ií]tulo)', str(c), re.I):
                name_col = c; break
        for c in cat_df.columns:
            if re.search(r'(desc|detalle|enunciado|texto)', str(c), re.I):
                desc_col = c; break
        if name_col is None: name_col = cat_df.columns[0]
        if desc_col is None: desc_col = name_col
        out = cat_df[[name_col, desc_col]].copy()
        out.columns = ["Objetivo", "DescripcionObjetivo"]
        out["Objetivo"] = out["Objetivo"].astype(str).str.strip()
        out["DescripcionObjetivo"] = out["DescripcionObjetivo"].astype(str)
        out = out.drop_duplicates(subset=["Objetivo"]).reset_index(drop=True)
        return out

    vals = []
    if col_obj_actual and col_obj_actual in df.columns:
        vals.append(df[col_obj_actual].dropna().astype(str))
    if col_obj_sug and col_obj_sug in df.columns:
        vals.append(df[col_obj_sug].dropna().astype(str))
    if not vals:
        return pd.DataFrame({"Objetivo": [], "DescripcionObjetivo": []})
    objetivos = pd.concat(vals).unique()
    cat = pd.DataFrame({"Objetivo": objetivos})
    cat["DescripcionObjetivo"] = cat["Objetivo"]
    return cat

def jaccard_matrix(A, B):
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            inter = len(a & b)
            union = len(a | b) or 1
            M[i, j] = inter / union
    return M

def score_and_recommend(df, obj_catalog, col_act, col_obj_actual, col_obj_sug,
                        w_name=0.35, w_profA=0.30, w_profS=0.25, w_overlap=0.10, balance_lambda=0.12):
    df = df.copy()
    if col_act in df.columns: df[col_act] = df[col_act].astype(str).fillna("")
    if col_obj_actual and col_obj_actual in df.columns: df[col_obj_actual] = df[col_obj_actual].astype(str).fillna("")
    if col_obj_sug and col_obj_sug in df.columns: df[col_obj_sug] = df[col_obj_sug].astype(str).fillna("")

    obj_catalog = obj_catalog.copy()
    obj_catalog["Objetivo"] = obj_catalog["Objetivo"].astype(str)
    obj_catalog["DescripcionObjetivo"] = obj_catalog["DescripcionObjetivo"].astype(str)
    objetivos = obj_catalog["Objetivo"].tolist()

    # Perfiles robustos (si falta Obj. actual o sugerido, se generan perfiles vacíos)
    if SKLEARN_OK:
        if col_obj_actual and col_obj_actual in df.columns:
            perfil_actual = {o: " ".join([normalize_text(t) for t in df.loc[df[col_obj_actual]==o, col_act].astype(str).tolist()]) for o in objetivos}
        else:
            perfil_actual = {o: "" for o in objetivos}
        if col_obj_sug and col_obj_sug in df.columns:
            perfil_sugerido = {o: " ".join([normalize_text(t) for t in df.loc[df[col_obj_sug]==o, col_act].astype(str).tolist()]) for o in objetivos}
        else:
            perfil_sugerido = {o: "" for o in objetivos}

        docs_activ = [normalize_text(t) for t in df[col_act].tolist()]
        docs_names = [normalize_text(o) for o in objetivos]
        docs_profA = [perfil_actual[o] for o in objetivos]
        docs_profS = [perfil_sugerido[o] for o in objetivos]

        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
        X = vectorizer.fit_transform(docs_activ + docs_names + docs_profA + docs_profS)
        n_act = len(docs_activ); n_obj = len(docs_names)
        X_act = X[:n_act]
        X_name = X[n_act:n_act+n_obj]
        X_profA = X[n_act+n_obj:n_act+2*n_obj]
        X_profS = X[n_act+2*n_obj:n_act+3*n_obj]
        sim_name = cosine_similarity(X_act, X_name)
        sim_profA = cosine_similarity(X_act, X_profA)
        sim_profS = cosine_similarity(X_act, X_profS)
    else:
        act_tokens = [set(tokenize(t)) for t in df[col_act].tolist()]
        obj_tokens = [set(tokenize(o)) for o in objetivos]
        sim_name = jaccard_matrix(act_tokens, obj_tokens)
        sim_profA = np.zeros_like(sim_name)
        sim_profS = np.zeros_like(sim_name)

    overlap_to_name = np.zeros_like(sim_name)
    for j, o in enumerate(objetivos):
        for i, a in enumerate(df[col_act]):
            overlap_to_name[i, j] = keyword_overlap_score(a, o)

    if col_obj_actual and col_obj_actual in df.columns:
        counts_actual = df[col_obj_actual].value_counts().reindex(objetivos, fill_value=0)
    else:
        counts_actual = pd.Series(0, index=objetivos)
    max_c = counts_actual.max() if counts_actual.max() > 0 else 1
    penalty = balance_lambda * (counts_actual.values / max_c)

    base = (w_name*sim_name) + (w_profA*sim_profA) + (w_profS*sim_profS) + (w_overlap*overlap_to_name)
    final_scores = base - penalty

    top_idx = final_scores.argmax(axis=1)
    top_scores = final_scores[np.arange(final_scores.shape[0]), top_idx]
    if final_scores.shape[1] > 1:
        second_idx = np.argsort(-final_scores, axis=1)[:, 1]
        second_scores = final_scores[np.arange(final_scores.shape[0]), second_idx]
    else:
        second_idx = np.full(final_scores.shape[0], -1); second_scores = np.zeros(final_scores.shape[0])

    def conf(a, b):
        if a <= 0: return 0.0
        return float((a - b) / a)
    confidences = np.array([conf(t, s) for t, s in zip(top_scores, second_scores)])

    drivers = [top_overlap_terms(df.iloc[i][col_act], objetivos[top_idx[i]]) for i in range(len(df))]

    out = pd.DataFrame({
        "Actividad": df[col_act],
        "Objetivo_actual": df[col_obj_actual] if (col_obj_actual and col_obj_actual in df.columns) else "",
        "Objetivo_sugerido_previo": df[col_obj_sug] if (col_obj_sug and col_obj_sug in df.columns) else "",
        "Objetivo_sugerido_mejorado": [objetivos[k] for k in top_idx],
        "Segundo_mejor_objetivo": [objetivos[k] if k != -1 else "" for k in second_idx],
        "Puntaje_mejorado": np.round(top_scores, 4),
        "Puntaje_segundo": np.round(second_scores, 4),
        "Confianza_%": np.round(confidences * 100, 1),
        "Coincidencias_clave": drivers
    })
    p = out["Puntaje_mejorado"].values
    p_min, p_max = float(np.min(p)), float(np.max(p))
    out["Consistencia_estimada_%"] = np.round((p - p_min) / (p_max - p_min + 1e-9) * 100, 1)

    resumen_actual = df[col_obj_actual].value_counts().rename_axis("Objetivo").reset_index(name="N_actual") if (col_obj_actual and col_obj_actual in df.columns) else pd.DataFrame(columns=["Objetivo","N_actual"])
    resumen_prev = df[col_obj_sug].value_counts().rename_axis("Objetivo").reset_index(name="N_sugerido_prev") if (col_obj_sug and col_obj_sug in df.columns) else pd.DataFrame(columns=["Objetivo","N_sugerido_prev"])
    resumen_mej = out["Objetivo_sugerido_mejorado"].value_counts().rename_axis("Objetivo").reset_index(name="N_sugerido_mejorado")
    base_df = obj_catalog[["Objetivo"]].copy() if not obj_catalog.empty else resumen_mej[["Objetivo"]].copy()
    resumen = base_df.merge(resumen_actual, on="Objetivo", how="left")\
                     .merge(resumen_prev, on="Objetivo", how="left")\
                     .merge(resumen_mej, on="Objetivo", how="left")\
                     .fillna(0)
    for c in ["N_actual","N_sugerido_prev","N_sugerido_mejorado"]:
        if c in resumen.columns: resumen[c] = resumen[c].astype(int)
    if "N_sugerido_prev" in resumen.columns:
        resumen["Delta_vs_prev"] = resumen["N_sugerido_mejorado"] - resumen["N_sugerido_prev"]
    if "N_actual" in resumen.columns:
        resumen["Delta_vs_actual"] = resumen["N_sugerido_mejorado"] - resumen["N_actual"]

    discrepancias = out.loc[out["Objetivo_sugerido_mejorado"] != out["Objetivo_sugerido_previo"]].copy()
    return out, resumen, discrepancias

def to_excel_bytes(propuestas, resumen, discrepancias, obj_catalog):
    import openpyxl
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        propuestas.to_excel(writer, index=False, sheet_name="Propuestas")
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        discrepancias.to_excel(writer, index=False, sheet_name="Discrepancias")
        obj_catalog.to_excel(writer, index=False, sheet_name="Objetivos_catalogo")
    buf.seek(0)
    return buf
