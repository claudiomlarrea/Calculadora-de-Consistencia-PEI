import io
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

from utils import (
    normalize_colnames, clean_rows, detect_columns, 
    compute_pairwise_consistency_single,
    parse_pei_pdf, build_plan_index, compute_consistency_pei_single,
    excel_from_blocks, normalize_text, has_real_activity, has_objective_assigned,
    count_all_activity_cells, count_activity_objective_pairs
)
from rapidfuzz import fuzz

st.set_page_config(page_title="Análisis de Consistencia PEI", layout="wide")
st.title("Análisis de Consistencia - Formulario Único PEI")

# ==================== FUNCIONES DE ANÁLISIS ====================

def generate_advanced_report(df: pd.DataFrame, col_obj: str, col_act: str, 
                           scores_col: str = 'score_obj_vs_act',
                           classification_col: str = 'clasificacion_calculada') -> Dict[str, Any]:
    """Genera un informe avanzado basado en actividades individuales"""
    report = {}
    
    if len(df) == 0:
        return {
            'estadisticas_generales': {'total_actividades': 0, 'promedio_consistencia': 0, 'mediana': 0, 'p25': 0, 'p75': 0, 'minimo': 0, 'maximo': 0},
            'distribucion_niveles': {'alta_75_plus': 0, 'media_50_74': 0, 'baja_menor_50': 0},
            'duplicados': pd.DataFrame(),
            'recomendaciones': {'corto_plazo': [], 'mediano_plazo': [], 'largo_plazo': []}
        }
    
    # 1. Estadísticas generales
    total_activities = len(df)
    scores = df[scores_col].astype(float)
    
    report['estadisticas_generales'] = {
        'total_actividades': total_activities,
        'promedio_consistencia': round(scores.mean(), 1),
        'mediana': round(scores.median(), 1),
        'p25': round(scores.quantile(0.25), 1),
        'p75': round(scores.quantile(0.75), 1),
        'minimo': round(scores.min(), 1),
        'maximo': round(scores.max(), 1)
    }
    
    # 2. Distribución por niveles
    classification_counts = df[classification_col].value_counts()
    report['distribucion_niveles'] = {
        'alta_75_plus': int(classification_counts.get('plena', 0)),
        'media_50_74': int(classification_counts.get('parcial', 0)),
        'baja_menor_50': int(classification_counts.get('nula', 0))
    }
    
    # 3. Actividades duplicadas
    report['duplicados'] = find_duplicate_activities(df, col_act)
    
    # 4. Recomendaciones
    report['recomendaciones'] = generate_recommendations(report)
    
    return report

def find_duplicate_activities(df: pd.DataFrame, col_act: str, threshold: float = 85.0) -> pd.DataFrame:
    """Encuentra actividades duplicadas o muy similares"""
    
    if len(df) == 0:
        return pd.DataFrame()
    
    try:
        activities = df[col_act].dropna().astype(str).tolist()
        normalized_activities = [normalize_text(act) for act in activities]
        
        duplicates_info = defaultdict(list)
        processed = set()
        
        for i, (act1, norm1) in enumerate(zip(activities, normalized_activities)):
            if i in processed or not norm1:
                continue
                
            similar_group = [act1]
            processed.add(i)
            
            for j, (act2, norm2) in enumerate(zip(activities, normalized_activities)):
                if i >= j or j in processed or not norm2:
                    continue
                    
                similarity = fuzz.token_set_ratio(norm1, norm2)
                if similarity >= threshold:
                    similar_group.append(act2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                representative = similar_group[0][:60] + "..." if len(similar_group[0]) > 60 else similar_group[0]
                duplicates_info[representative] = len(similar_group)
        
        duplicates_df = pd.DataFrame([
            {'Actividad_normalizada': act, 'Repeticiones': count}
            for act, count in duplicates_info.items()
        ])
        
        return duplicates_df.sort_values('Repeticiones', ascending=False) if not duplicates_df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def generate_recommendations(report: Dict[str, Any]) -> Dict[str, List[str]]:
    """Genera recomendaciones basadas en el análisis"""
    
    recommendations = {
        'corto_plazo': [],
        'mediano_plazo': [],
        'largo_plazo': []
    }
    
    stats = report['estadisticas_generales']
    duplicates = report['duplicados']
    
    # Recomendaciones de corto plazo
    if stats['promedio_consistencia'] < 30:
        recommendations['corto_plazo'].append(
            "Implementar plantillas estandarizadas para redacción de actividades"
        )
        recommendations['corto_plazo'].append(
            "Crear glosario de verbos operativos y entregables específicos"
        )
    
    if not duplicates.empty:
        recommendations['corto_plazo'].append(
            f"Revisar y consolidar {len(duplicates)} grupos de actividades similares"
        )
    
    # Recomendaciones de mediano plazo
    recommendations['mediano_plazo'].append(
        "Establecer trazabilidad KPI/evidencias para cada objetivo específico"
    )
    
    recommendations['mediano_plazo'].append(
        "Revisar objetivos específicos con consistencias más bajas"
    )
    
    # Recomendaciones de largo plazo
    recommendations['largo_plazo'].append(
        "Implementar revisión trimestral automatizada de consistencias"
    )
    recommendations['largo_plazo'].append(
        "Establecer comité de gobernanza del PEI para seguimiento continuo"
    )
    
    return recommendations

def format_report_for_display(report: Dict[str, Any], cell_counts: Dict[str, int]) -> str:
    """Formatea el reporte para mostrar en Streamlit"""
    
    stats = report['estadisticas_generales']
    dist = report['distribucion_niveles']
    
    formatted = f"""
# Informe de Consistencia PEI

## Resumen del Formulario
- **Total personas**: {cell_counts['total_cells']} participantes
- **Actividades propuestas**: {stats['total_actividades']} actividades reales
- **Celdas "None"**: {cell_counts['none_values']} (sin propuesta para ese objetivo)
- **Celdas vacías**: {cell_counts['empty_cells']}

## Estadísticas de Consistencia
- **Promedio de consistencia**: {stats['promedio_consistencia']}%
- **Mediana**: {stats['mediana']}% | **P25**: {stats['p25']}% | **P75**: {stats['p75']}%
- **Rango**: {stats['minimo']}% - {stats['maximo']}%

## Distribución por Niveles
- **Alta (≥75%)**: {dist['alta_75_plus']} actividades
- **Media (50-74%)**: {dist['media_50_74']} actividades  
- **Baja (<50%)**: {dist['baja_menor_50']} actividades

"""
    
    if dist['baja_menor_50'] > dist['alta_75_plus'] + dist['media_50_74']:
        formatted += "⚠️ **Interpretación**: Alta concentración en nivel Bajo sugiere redacciones genéricas u objetivos poco acotados.\n\n"
    
    return formatted

# ==================== APLICACIÓN PRINCIPAL ====================

# Carga del archivo único
uploaded = st.file_uploader("Subí el **Formulario Único** (XLSX o CSV)", type=["xlsx","csv"])
if not uploaded:
    st.stop()

# leer archivo
bio = io.BytesIO(uploaded.getvalue()); bio.name = uploaded.name
if uploaded.name.lower().endswith(".xlsx"):
    df = pd.read_excel(bio, engine="openpyxl")
else:
    try:
        df = pd.read_csv(bio, encoding="utf-8", sep=None, engine="python")
    except Exception:
        bio.seek(0); df = pd.read_csv(bio, encoding="latin-1", sep=None, engine="python")

df = normalize_colnames(df)
df_original_size = len(df)
df = clean_rows(df)

# detectar columnas y permitir cambio
obj_default, act_default = detect_columns(df)
st.subheader("Selecciona columnas")
c1, c2 = st.columns(2)
with c1:
    col_obj = st.selectbox("Columna de **Objetivo específico**", options=list(df.columns), 
                          index=(list(df.columns).index(obj_default) if obj_default in df.columns else 0))
with c2:
    col_act = st.selectbox("Columna de **Actividad**", options=list(df.columns), 
                          index=(list(df.columns).index(act_default) if act_default in df.columns else (1 if len(df.columns)>1 else 0)))

# DIAGNÓSTICO CORRECTO
st.subheader("Diagnóstico del Formulario")

# Contar por celdas individuales
cell_counts = count_all_activity_cells(df, col_act)
pair_counts = count_activity_objective_pairs(df, col_obj, col_act)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Participantes", cell_counts['total_cells'])
with col2:
    st.metric("Actividades Reales", cell_counts['real_activities'])
with col3:
    st.metric("Sin Propuesta ('None')", cell_counts['none_values'])
with col4:
    st.metric("Celdas Vacías", cell_counts['empty_cells'])

# Explicación del formulario
st.info(f"""
📋 **Estructura del Formulario**:
- **{cell_counts['total_cells']} personas** completaron el formulario
- **Cada persona** puede proponer actividades para algunos objetivos específicos
- **"None"** significa "esta persona no propuso actividad para este objetivo"
- **Se analizan**: {cell_counts['real_activities']} actividades reales propuestas
""")

# Análisis detallado por tipos
st.write("### Análisis por Tipo de Celda")

col1, col2 = st.columns(2)

with col1:
    st.write("**Actividades que se ANALIZAN:**")
    real_activities = df[df[col_act].apply(has_real_activity)]
    if not real_activities.empty:
        st.write(f"Total: {len(real_activities)} actividades reales")
        st.dataframe(real_activities[[col_act]].head(8), use_container_width=True)
    else:
        st.warning("No se encontraron actividades reales")

with col2:
    st.write("**Celdas que NO se analizan:**")
    non_activities = df[~df[col_act].apply(has_real_activity)]
    if not non_activities.empty:
        st.write(f"Total: {len(non_activities)} celdas")
        # Mostrar tipos de valores no-actividad
        none_count = non_activities[col_act].apply(lambda x: str(x).strip().lower() == 'none').sum()
        empty_count = len(non_activities) - none_count
        
        st.write(f"- 'None' (sin propuesta): {none_count}")
        st.write(f"- Vacías/NaN: {empty_count}")
        
        st.dataframe(non_activities[[col_act]].head(5), use_container_width=True)

# Relación actividades-objetivos
st.write("### Relación Actividades-Objetivos")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ambos Completos", pair_counts['both_complete'])
with col2:
    st.metric("Solo Actividad", pair_counts['activity_only'])
with col3:
    st.metric("Solo Objetivo", pair_counts['objective_only'])
with col4:
    st.metric("Ambos Vacíos", pair_counts['both_empty'])

st.subheader("Previsualización")
st.dataframe(df[[col_obj, col_act]].head(12), use_container_width=True)

# Umbrales
st.subheader("Umbrales")
c1, c2 = st.columns(2)
with c1:
    t_plena = st.slider("Umbral 'plena'", 70, 100, 88, 1)
with c2:
    t_parcial = st.slider("Umbral 'parcial'", 50, 90, 68, 1)
thr = {"plena": float(t_plena), "parcial": float(t_parcial)}

# Análisis de consistencia
if st.button("Realizar Análisis de Consistencia", type="primary"):
    with st.spinner("Analizando actividades individuales..."):
        sum_indep, det_indep = compute_pairwise_consistency_single(df, uploaded.name, col_obj, col_act, thr)
        
        # Verificar resultados
        st.success(f"✅ Analizadas {sum_indep['Total actividades']} actividades reales de {cell_counts['total_cells']} participantes")
        
        # Generar análisis avanzado
        advanced_report = generate_advanced_report(det_indep, col_obj, col_act)
        
        # Mostrar resumen ejecutivo
        st.markdown("---")
        st.markdown(format_report_for_display(advanced_report, cell_counts))
        
        # Actividades duplicadas
        st.subheader("Actividades Duplicadas o Similares")
        if not advanced_report['duplicados'].empty:
            st.dataframe(advanced_report['duplicados'], use_container_width=True)
            st.caption("**Sugerencia**: Consolidar duplicadas como líneas de trabajo con sub-tareas medibles")
        else:
            st.success("No se detectaron actividades duplicadas")
        
        # Recomendaciones
        st.subheader("Plan de Mejora")
        recs = advanced_report['recomendaciones']
        
        with st.expander("Corto Plazo (0-30 días)", expanded=True):
            for rec in recs['corto_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("Mediano Plazo (1-3 meses)"):
            for rec in recs['mediano_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("Largo Plazo (3+ meses)"):
            for rec in recs['largo_plazo']:
                st.write(f"• {rec}")
        
        # Descargas
        st.subheader("Descargas")
        
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar bloques para Excel
        blocks = [
            ("Resumen_Formulario", pd.DataFrame([cell_counts])),
            ("Estadisticas_Consistencia", pd.DataFrame([advanced_report['estadisticas_generales']])),
            ("Distribucion_Niveles", pd.DataFrame([advanced_report['distribucion_niveles']])),
            ("Actividades_Duplicadas", advanced_report['duplicados']),
            ("Detalle_Actividades", det_indep),
        ]
        
        # Generar Excel
        excel_bytes = excel_from_blocks(blocks)
        
        st.download_button(
            "Descargar Análisis Excel Completo", 
            data=excel_bytes, 
            file_name=f"analisis_consistencia_formulario_{ts}.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Análisis contra el PEI (opcional)
st.subheader("Análisis contra el PEI (opcional)")
pei_pdf = st.file_uploader("Subí el PDF del PEI (opcional)", type=["pdf"], key="pei")
if pei_pdf:
    with st.spinner("Analizando actividades contra PEI..."):
        pei = parse_pei_pdf(pei_pdf)
        index = build_plan_index(pei)
    sum_pei, det_pei = compute_consistency_pei_single(df, uploaded.name, col_act, index, thr)
    
    st.success(f"Análisis contra PEI completado: {sum_pei['Total actividades']} actividades evaluadas")
    
    # Mostrar resultados PEI
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consistencia Plena", sum_pei['Consistencia plena'], 
                 f"{sum_pei['Consistencia plena']/sum_pei['Total actividades']*100:.1f}%")
    with col2:
        st.metric("Consistencia Parcial", sum_pei['Consistencia parcial'], 
                 f"{sum_pei['Consistencia parcial']/sum_pei['Total actividades']*100:.1f}%")
    with col3:
        st.metric("Consistencia Nula", sum_pei['Consistencia nula'], 
                 f"{sum_pei['Consistencia nula']/sum_pei['Total actividades']*100:.1f}%")
