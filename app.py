
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
from datetime import datetime

from utils import (
    parse_uploaded_files,
    build_all_acts_and_summaries,
    build_excel_bytes,
    build_word_bytes,
)

st.set_page_config(page_title="Calculadora Consistencia PEI UCCuyo", layout="wide")

st.title("Calculadora de Consistencia con el PEI 2023–2027 (UCCuyo)")
st.caption("Subí las 6 tablas de actividades (Objetivos 1 a 6). La app genera automáticamente el Excel y el Informe Word.")

with st.expander("Instrucciones de carga", expanded=True):
    st.markdown("""
    - Subí **exactamente 6 archivos CSV**, uno por objetivo (1 a 6).
    - No importa el orden ni el nombre del archivo; la calculadora detecta el objetivo **por las columnas** internas.
    - Campos esperados por archivo: `AÑO`, `Objetivos específicos X`, `Actividades Objetivo X`, `Detalle de la Actividad Objetivo X`, `Unidad Académica o Administrativa` (con o sin espacios iniciales).
    - Se ignoran filas sin actividad real (marcadores con “-”).
    """)

uploaded = st.file_uploader(
    "Arrastrá y soltá aquí los 6 CSV",
    type=["csv"],
    accept_multiple_files=True,
    help="Podés seleccionar múltiples archivos a la vez.",
)

if uploaded:
    if len(uploaded) != 6:
        st.warning("Se detectaron **{}** archivos. Subí **exactamente 6** (Objetivos 1 a 6).".format(len(uploaded)))
    else:
        with st.spinner("Procesando archivos…"):
            try:
                raw_by_obj = parse_uploaded_files(uploaded)  # dict {1..6: df}
                if set(raw_by_obj.keys()) != set(range(1,7)):
                    st.error("No se pudieron identificar todos los objetivos (1 a 6) a partir de las columnas. Verificá que cada CSV tenga las columnas estándar.")
                else:
                    all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio = build_all_acts_and_summaries(raw_by_obj)

                    st.success("¡Listo! Se analizaron {} actividades.".format(len(all_acts)))

                    # Vistas
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Porcentaje por objetivo")
                        st.dataframe(pivot_summary, use_container_width=True)
                    with c2:
                        st.subheader("Cantidades por objetivo")
                        st.dataframe(pivot_counts, use_container_width=True)

                    # Gráfico
                    st.subheader("Consistencia con PEI por Objetivo (%)")
                    consistent = pivot_summary[["Objetivo", "Consistente con PEI (objetivo específico identificado)"]].copy()
                    consistent = consistent.sort_values("Objetivo")
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.bar(consistent["Objetivo"].astype(int), consistent["Consistente con PEI (objetivo específico identificado)"])
                    ax.set_xlabel("Objetivo")
                    ax.set_ylabel("Porcentaje de actividades consistentes")
                    ax.set_title("Consistencia con PEI por Objetivo (%)")
                    st.pyplot(fig, clear_figure=True)

                    st.subheader("Detalle de actividades (con % de consistencia por objetivo)")
                    st.dataframe(all_acts, use_container_width=True)

                    # Descargas
                    xlsx_bytes = build_excel_bytes(all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio)
                    st.download_button(
                        label="⬇️ Descargar Excel (consistencia_por_objetivo.xlsx)",
                        data=xlsx_bytes,
                        file_name="consistencia_por_objetivo.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                    docx_bytes = build_word_bytes(all_acts, pivot_summary, pivot_counts, dist_codigo, top_codigos, unidades_desvio, fig)
                    st.download_button(
                        label="⬇️ Descargar Informe Word (Informe_consistencia_PEI_UCCuyo.docx)",
                        data=docx_bytes,
                        file_name="Informe_consistencia_PEI_UCCuyo.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

                    with st.expander("Tablas auxiliares"):
                        st.write("Top códigos por objetivo")
                        st.dataframe(top_codigos, use_container_width=True)
                        st.write("Unidades con mayor número de desvíos")
                        st.dataframe(unidades_desvio, use_container_width=True)

            except Exception as e:
                st.exception(e)
else:
    st.info("Esperando que subas los 6 CSV…")
