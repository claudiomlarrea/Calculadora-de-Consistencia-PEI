
import io
import pandas as pd
import streamlit as st

from utilis import (
    maybe_from_formulario_unico,
    detect_columns, build_objective_catalog,
    score_and_recommend, to_excel_bytes
)

st.set_page_config(page_title="Calculadora de consistencia PEI – Propuestas mejoradas", layout="wide")
st.title("Calculadora de consistencia PEI – Propuestas de objetivo (versión mejorada)")
st.caption("Ahora acepta **Formulario Único crudo** (ancho). Si lo detecta, lo transforma automáticamente al formato estándar.")

with st.sidebar:
    st.header("Parámetros de modelo")
    w_name = st.slider("Peso: similitud con nombre del objetivo", 0.0, 1.0, 0.35, 0.05)
    w_profA = st.slider("Peso: perfil de actividades actuales", 0.0, 1.0, 0.30, 0.05)
    w_profS = st.slider("Peso: perfil de sugeridas previas", 0.0, 1.0, 0.25, 0.05)
    w_overlap = st.slider("Peso: solapamiento léxico", 0.0, 1.0, 0.10, 0.05)
    balance_lambda = st.slider("Penalización por sobre-asignación (λ)", 0.0, 0.5, 0.12, 0.01)
    st.markdown("---")
    thr_auto = st.slider("Umbral Auto (%)", 0, 100, 40, 1)
    thr_review = st.slider("Umbral Revisar (%)", 0, 100, 20, 1)

st.subheader("1) Cargar archivo de actividades (Excel)")
file = st.file_uploader("Subí: Formulario Único crudo **o** el Excel ya estandarizado con columnas 'Actividad' y 'Obj. actual'.", type=["xlsx","xls"])
st.subheader("2) (Opcional) Catálogo de objetivos")
cat_file = st.file_uploader("Catálogo (Objetivo, Descripción)", type=["xlsx","xls","csv"], key="cat")

if file is not None:
    df0 = pd.read_excel(file)
    # Si es Formulario Único (ancho), convertir automáticamente
    df = maybe_from_formulario_unico(df0)
    converted = (df is not df0) or (set(df.columns) != set(df0.columns))
    if converted:
        st.success("Detecté el **Formulario Único** y lo convertí automáticamente a formato estándar.")

    col_act, col_obj_actual, col_obj_sug = detect_columns(df)
    if col_act is None:
        st.error("No pude detectar la columna de actividad. Subí un Formulario Único válido o un Excel con 'Actividad'.")
        st.stop()
    st.info(f"Detectado → Actividad: **{col_act}** | Obj. actual: **{col_obj_actual}** | Obj. sugerido: **{col_obj_sug}**")

    # Catálogo
    cat_df = None
    if cat_file is not None:
        try:
            if cat_file.name.lower().endswith(".csv"):
                cat_df = pd.read_csv(cat_file)
            else:
                cat_df = pd.read_excel(cat_file)
        except Exception as e:
            st.warning(f"No pude leer el catálogo: {e}")
    obj_catalog = build_objective_catalog(df, col_obj_actual, col_obj_sug, cat_df=cat_df)
    if obj_catalog.empty:
        st.error("No pude construir el catálogo de objetivos (no encontré nombres de objetivos).")
        st.stop()

    propuestas, resumen, discrepancias = score_and_recommend(
        df, obj_catalog, col_act, col_obj_actual, col_obj_sug,
        w_name=w_name, w_profA=w_profA, w_profS=w_profS, w_overlap=w_overlap, balance_lambda=balance_lambda
    )
    def etiqueta(c): return "Auto" if c>=thr_auto else ("Revisar" if c>=thr_review else "Validación")
    propuestas["Decisión"] = propuestas["Confianza_%"].apply(etiqueta)

    st.subheader("Resultados")
    st.dataframe(propuestas.head(50), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Discrepancias**")
        st.dataframe(discrepancias, use_container_width=True, height=320)
    with cb:
        st.markdown("**Resumen por objetivo**")
        st.dataframe(resumen, use_container_width=True, height=320)

    buf = to_excel_bytes(propuestas, resumen, discrepancias, obj_catalog)
    st.download_button("Descargar Excel", buf, file_name="Propuestas_objetivo_mejoradas.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.warning("Esperando archivo de actividades…")
