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
    normalize_colnames, clean_rows, detect_columns, count_valid_pairs, 
    compute_pairwise_consistency_single,
    parse_pei_pdf, build_plan_index, compute_consistency_pei_single,
    excel_from_blocks, normalize_text, is_empty_value, is_meaningful_text
)
from rapidfuzz import fuzz

st.set_page_config(page_title="Análisis Avanzado de Consistencia PEI", layout="wide")
st.title("Análisis Avanzado de Consistencia - Formulario Único PEI")

# ==================== FUNCIONES AUXILIARES ====================

def count_all_activities(df: pd.DataFrame, col_act: str) -> int:
    """Cuenta todas las actividades con cualquier contenido"""
    return int(df[col_act].apply(lambda x: is_meaningful_text(x, min_length=1)).sum())

def count_with_objective(df: pd.DataFrame, col_obj: str) -> int:
    """Cuenta filas con objetivo asignado"""
    return int(df[col_obj].apply(lambda x: is_meaningful_text(x, min_length=1)).sum())

# ==================== FUNCIONES DE ANÁLISIS AVANZADO ====================

def generate_advanced_report(df: pd.DataFrame, col_obj: str, col_act: str, 
                           scores_col: str = 'score_obj_vs_act',
                           classification_col: str = 'clasificacion_calculada') -> Dict[str, Any]:
    """Genera un informe avanzado similar al documento de conclusiones"""
    report = {}
    
    # 1. Estadísticas generales
    valid_activities = df[df[col_act].apply(lambda x: is_meaningful_text(x, min_length=1))]
    total_activities = len(valid_activities)
    
    if total_activities == 0:
        # Retornar reporte vacío si no hay datos
        return {
            'estadisticas_generales': {'total_actividades': 0, 'promedio_consistencia': 0, 'mediana': 0, 'p25': 0, 'p75': 0, 'minimo': 0, 'maximo': 0},
            'distribucion_niveles': {'alta_75_plus': 0, 'media_50_74': 0, 'baja_menor_50': 0},
            'rendimiento_objetivos': pd.DataFrame(),
            'dispersion_objetivos': pd.DataFrame(),
            'potencial_mejora': pd.DataFrame(),
            'duplicados': pd.DataFrame(),
            'sin_objetivo': {'total_sin_objetivo': 0, 'porcentaje_sin_objetivo': 0, 'ejemplos': []},
            'recomendaciones': {'corto_plazo': [], 'mediano_plazo': [], 'largo_plazo': []}
        }
    
    scores = valid_activities[scores_col].astype(float)
    
    report['estadisticas_generales'] = {
        'total_actividades': total_activities,
        'promedio_consistencia': round(scores.mean(), 1) if len(scores) > 0 else 0,
        'mediana': round(scores.median(), 1) if len(scores) > 0 else 0,
        'p25': round(scores.quantile(0.25), 1) if len(scores) > 0 else 0,
        'p75': round(scores.quantile(0.75), 1) if len(scores) > 0 else 0,
        'minimo': round(scores.min(), 1) if len(scores) > 0 else 0,
        'maximo': round(scores.max(), 1) if len(scores) > 0 else 0
    }
    
    # 2. Distribución por niveles
    classification_counts = valid_activities[classification_col].value_counts() if len(valid_activities) > 0 else pd.Series()
    report['distribucion_niveles'] = {
        'alta_75_plus': int(classification_counts.get('plena', 0)),
        'media_50_74': int(classification_counts.get('parcial', 0)),
        'baja_menor_50': int(classification_counts.get('nula', 0))
    }
    
    # 3. Análisis de actividades sin objetivo
    missing_objectives = analyze_missing_objectives(df, col_obj, col_act)
    report['sin_objetivo'] = missing_objectives
    
    # 4. Otros análisis (simplificados para evitar errores)
    report['rendimiento_objetivos'] = pd.DataFrame()
    report['dispersion_objetivos'] = pd.DataFrame()
    report['potencial_mejora'] = pd.DataFrame()
    report['duplicados'] = find_duplicate_activities(valid_activities, col_act)
    
    # 5. Recomendaciones
    recommendations = generate_recommendations(report)
    report['recomendaciones'] = recommendations
    
    return report

def analyze_missing_objectives(df: pd.DataFrame, col_obj: str, col_act: str) -> Dict[str, Any]:
    """Analiza actividades sin objetivo asignado"""
    
    missing_obj = df[df[col_obj].apply(lambda x: not is_meaningful_text(x, min_length=1)) & 
                     df[col_act].apply(lambda x: is_meaningful_text(x, min_length=1))]
    
    return {
        'total_sin_objetivo': len(missing_obj),
        'porcentaje_sin_objetivo': round(len(missing_obj) / len(df) * 100, 1) if len(df) > 0 else 0,
        'ejemplos': missing_obj[col_act].head(5).tolist() if len(missing_obj) > 0 else []
    }

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
                representative = similar_group[0][:50] + "..." if len(similar_group[0]) > 50 else similar_group[0]
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
    sin_objetivo = report['sin_objetivo']
    
    # Recomendaciones de corto plazo
    if stats['promedio_consistencia'] < 30:
        recommendations['corto_plazo'].append(
            "Implementar higiene de redacción con plantillas estandarizadas para actividades"
        )
        recommendations['corto_plazo'].append(
            "Crear glosario de verbos operativos y entregables específicos"
        )
    
    if not duplicates.empty:
        recommendations['corto_plazo'].append(
            f"Consolidar {len(duplicates)} grupos de actividades duplicadas identificadas"
        )
    
    if sin_objetivo['total_sin_objetivo'] > 0:
        recommendations['corto_plazo'].append(
            f"Asignar objetivos específicos a {sin_objetivo['total_sin_objetivo']} actividades sin clasificar"
        )
    
    # Recomendaciones de mediano plazo
    recommendations['mediano_plazo'].append(
        "Establecer trazabilidad KPI/evidencias para cada objetivo específico"
    )
    
    recommendations['mediano_plazo'].append(
        "Revisar y reencuadrar objetivos específicos con baja consistencia"
    )
    
    # Recomendaciones de largo plazo
    recommendations['largo_plazo'].append(
        "Implementar revisión trimestral automatizada de consistencias"
    )
    recommendations['largo_plazo'].append(
        "Establecer comité de gobernanza del PEI para seguimiento continuo"
    )
    
    return recommendations

def format_report_for_display(report: Dict[str, Any]) -> str:
    """Formatea el reporte para mostrar en Streamlit"""
    
    stats = report['estadisticas_generales']
    dist = report['distribucion_niveles']
    sin_obj = report['sin_objetivo']
    
    formatted = f"""
# Informe Avanzado de Consistencia PEI

## Estadísticas Generales
- **Total actividades evaluadas**: {stats['total_actividades']}
- **Promedio de consistencia**: {stats['promedio_consistencia']}%
- **Mediana**: {stats['mediana']}% | **P25**: {stats['p25']}% | **P75**: {stats['p75']}%
- **Rango**: {stats['minimo']}% - {stats['maximo']}%

## Distribución por Niveles
- **Alta (≥75%)**: {dist['alta_75_plus']} actividades
- **Media (50-74%)**: {dist['media_50_74']} actividades  
- **Baja (<50%)**: {dist['baja_menor_50']} actividades

## Actividades sin Objetivo Asignado
- **Total sin objetivo**: {sin_obj['total_sin_objetivo']} actividades ({sin_obj['porcentaje_sin_objetivo']}%)

"""
    
    if dist['baja_menor_50'] > dist['alta_75_plus'] + dist['media_50_74']:
        formatted += "⚠️ **Interpretación**: Alta concentración en nivel Bajo sugiere redacciones genéricas u objetivos poco acotados.\n\n"
    
    if sin_obj['total_sin_objetivo'] > 20:
        formatted += "⚠️ **Atención**: Alto número de actividades sin objetivo específico asignado.\n\n"
    
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

# DIAGNÓSTICOS DETALLADOS
st.subheader("Diagnóstico Detallado de Datos")

total_meaningful = count_all_activities(df, col_act)
total_with_obj = count_with_objective(df, col_obj)
total_both = count_valid_pairs(df, col_obj, col_act)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filas Originales", df_original_size)
with col2:
    st.metric("Actividades Válidas", total_meaningful)
with col3:
    st.metric("Con Objetivo", total_with_obj)
with col4:
    st.metric("Completas", total_both)

# Análisis detallado de lo que se considera válido/inválido
st.write("### Análisis de Validez por Tipo")

# Mostrar ejemplos de cada categoría
col1, col2 = st.columns(2)

with col1:
    st.write("**✅ Actividades Válidas (se analizan):**")
    valid_activities = df[df[col_act].apply(lambda x: is_meaningful_text(x, min_length=1))]
    if not valid_activities.empty:
        st.write(f"Total: {len(valid_activities)} actividades")
        st.dataframe(valid_activities[[col_act]].head(5), use_container_width=True)
    
    # Mostrar actividades SIN objetivo
    st.write("**⚠️ Actividades sin Objetivo (se analizan con objetivo genérico):**")
    without_obj = valid_activities[~valid_activities[col_obj].apply(lambda x: is_meaningful_text(x, min_length=1))]
    if not without_obj.empty:
        st.write(f"Total: {len(without_obj)} actividades")
        st.dataframe(without_obj[[col_act]].head(3), use_container_width=True)
    else:
        st.success("Todas las actividades válidas tienen objetivo asignado")

with col2:
    st.write("**❌ Actividades Inválidas (se excluyen):**")
    invalid_activities = df[~df[col_act].apply(lambda x: is_meaningful_text(x, min_length=1))]
    if not invalid_activities.empty:
        st.write(f"Total: {len(invalid_activities)} registros excluidos")
        st.write("**Motivos de exclusión:**")
        
        # Análisis de por qué se excluyen
        exclusion_reasons = []
        for _, row in invalid_activities.head(10).iterrows():
            activity_val = row[col_act]
            if pd.isna(activity_val):
                exclusion_reasons.append("Valor nulo/NaN")
            elif str(activity_val).strip().lower() in ["none", "nan", "null", "", "-"]:
                exclusion_reasons.append(f"Valor vacío: '{activity_val}'")
            else:
                exclusion_reasons.append(f"Otro: '{str(activity_val)[:30]}...'")
        
        for reason in set(exclusion_reasons):
            count = exclusion_reasons.count(reason)
            st.write(f"- {reason}: {count} casos")
            
        st.dataframe(invalid_activities[[col_act]].head(3), use_container_width=True)
    else:
        st.success("Todas las filas contienen actividades válidas")

if total_meaningful != df_original_size:
    excluded_count = df_original_size - total_meaningful
    st.warning(f"⚠️ Se excluyeron {excluded_count} registros ({excluded_count/df_original_size*100:.1f}%) por no tener contenido válido en la columna de actividades.")

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
if st.button("Realizar Análisis Avanzado", type="primary"):
    with st.spinner("Calculando consistencias..."):
        sum_indep, det_indep = compute_pairwise_consistency_single(df, uploaded.name, col_obj, col_act, thr)
        
        # Generar análisis avanzado
        advanced_report = generate_advanced_report(det_indep, col_obj, col_act)
        
        # Mostrar resumen ejecutivo
        st.markdown("---")
        st.markdown(format_report_for_display(advanced_report))
        
        # Mostrar duplicados si existen
        st.subheader("Actividades Duplicadas o Similares")
        if not advanced_report['duplicados'].empty:
            st.dataframe(advanced_report['duplicados'], use_container_width=True)
            st.caption("**Sugerencia**: Consolidar duplicadas como líneas de trabajo con sub-tareas medibles")
        else:
            st.success("No se detectaron actividades duplicadas")
        
        # Actividades sin objetivo
        sin_obj_info = advanced_report['sin_objetivo']
        if sin_obj_info['total_sin_objetivo'] > 0:
            st.subheader("⚠️ Actividades sin Objetivo Específico")
            st.warning(f"Se encontraron **{sin_obj_info['total_sin_objetivo']}** actividades sin objetivo asignado ({sin_obj_info['porcentaje_sin_objetivo']}% del total)")
            
            if sin_obj_info['ejemplos']:
                with st.expander(f"Ver ejemplos de actividades sin objetivo"):
                    for i, ejemplo in enumerate(sin_obj_info['ejemplos'], 1):
                        st.write(f"{i}. {ejemplo}")
        
        # Recomendaciones
        st.subheader("Plan de Mejora por Etapas")
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
            ("Resumen_Estadisticas", pd.DataFrame([advanced_report['estadisticas_generales']])),
            ("Distribucion_Niveles", pd.DataFrame([advanced_report['distribucion_niveles']])),
            ("Sin_Objetivo", pd.DataFrame([advanced_report['sin_objetivo']])),
            ("Actividades_Duplicadas", advanced_report['duplicados']),
            ("Detalle_Completo", det_indep),
        ]
        
        # Generar Excel
        excel_bytes = excel_from_blocks(blocks)
        
        st.download_button(
            "Descargar Análisis Excel Completo", 
            data=excel_bytes, 
            file_name=f"analisis_consistencia_avanzado_{ts}.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Contra el PEI (opcional)
st.subheader("Análisis contra el PEI (opcional)")
pei_pdf = st.file_uploader("Subí el PDF del PEI (opcional)", type=["pdf"], key="pei")
if pei_pdf:
    with st.spinner("Analizando PEI..."):
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
