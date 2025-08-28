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
    normalize_colnames, clean_rows, detect_columns, count_valid_pairs, count_all_activities,
    compute_pairwise_consistency_single,
    parse_pei_pdf, build_plan_index, compute_consistency_pei_single,
    excel_from_blocks, normalize_text, is_empty_value, is_meaningful_text
)
from rapidfuzz import fuzz

st.set_page_config(page_title="Análisis Avanzado de Consistencia PEI", layout="wide")
st.title("📊 Análisis Avanzado de Consistencia - Formulario Único PEI")

# ==================== FUNCIONES DE ANÁLISIS AVANZADO ====================

def generate_advanced_report(df: pd.DataFrame, col_obj: str, col_act: str, 
                           scores_col: str = 'score_obj_vs_act',
                           classification_col: str = 'clasificacion_calculada') -> Dict[str, Any]:
    """Genera un informe avanzado similar al documento de conclusiones"""
    report = {}
    
    # 1. Estadísticas generales
    valid_activities = df[df[col_act].apply(lambda x: is_meaningful_text(x, min_length=3))]
    total_activities = len(valid_activities)
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
    
    # 3. Rendimiento por objetivo específico (solo si hay objetivos)
    obj_performance = analyze_objectives_performance(valid_activities, col_obj, scores_col, classification_col)
    report['rendimiento_objetivos'] = obj_performance
    
    # 4. Objetivos con mayor dispersión
    dispersion_analysis = analyze_objectives_dispersion(valid_activities, col_obj, scores_col)
    report['dispersion_objetivos'] = dispersion_analysis
    
    # 5. Actividades con potencial de mejora
    improvement_potential = find_improvement_opportunities(valid_activities, col_obj, col_act, scores_col)
    report['potencial_mejora'] = improvement_potential
    
    # 6. Actividades duplicadas
    duplicates = find_duplicate_activities(valid_activities, col_act)
    report['duplicados'] = duplicates
    
    # 7. Análisis de actividades sin objetivo
    missing_objectives = analyze_missing_objectives(df, col_obj, col_act)
    report['sin_objetivo'] = missing_objectives
    
    # 8. Recomendaciones
    recommendations = generate_recommendations(report)
    report['recomendaciones'] = recommendations
    
    return report

def analyze_objectives_performance(df: pd.DataFrame, col_obj: str, scores_col: str, 
                                 classification_col: str) -> pd.DataFrame:
    """Analiza el rendimiento por objetivo específico"""
    
    if len(df) == 0:
        return pd.DataFrame()
    
    def clean_objective_text(obj_text):
        if is_empty_value(obj_text):
            return "Sin objetivo asignado"
        text = str(obj_text).strip()
        match = re.match(r'^(\d+\.\d+)\s*[.-]*\s*(.*?)(?:\s*(?:Actividades|Resultados|Indicadores).*)?$', 
                        text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"{match.group(1)} {match.group(2).strip()}"
        return text[:100] + "..." if len(text) > 100 else text
    
    df_clean = df.copy()
    df_clean['objetivo_limpio'] = df_clean[col_obj].apply(clean_objective_text)
    
    # Verificar que tenemos datos para analizar
    if scores_col not in df_clean.columns or classification_col not in df_clean.columns:
        return pd.DataFrame()
    
    try:
        grouped = df_clean.groupby('objetivo_limpio').agg({
            scores_col: ['count', 'mean', 'median', 'std'],
            classification_col: lambda x: (x == 'nula').sum() / len(x) * 100
        }).round(1)
        
        grouped.columns = ['N_actividades', 'Promedio', 'Mediana', 'Desvio_std', 'Porc_en_Bajo']
        grouped = grouped.sort_values('Promedio').reset_index()
        
        # Calcular IQR
        iqr_values = []
        for obj in grouped['objetivo_limpio']:
            obj_data = df_clean[df_clean['objetivo_limpio'] == obj][scores_col]
            iqr = obj_data.quantile(0.75) - obj_data.quantile(0.25) if len(obj_data) > 1 else 0
            iqr_values.append(round(iqr, 1))
        
        grouped['IQR'] = iqr_values
        return grouped.head(10)
    except Exception:
        return pd.DataFrame()

def analyze_objectives_dispersion(df: pd.DataFrame, col_obj: str, scores_col: str) -> pd.DataFrame:
    """Analiza la dispersión interna de objetivos"""
    
    if len(df) == 0 or scores_col not in df.columns:
        return pd.DataFrame()
    
    def clean_objective_text(obj_text):
        if is_empty_value(obj_text):
            return "Sin objetivo asignado"
        text = str(obj_text).strip()
        match = re.match(r'^(\d+\.\d+)\s*[.-]*\s*(.*?)(?:\s*(?:Actividades|Resultados|Indicadores).*)?$', 
                        text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"{match.group(1)} {match.group(2).strip()}"
        return text[:100] + "..." if len(text) > 100 else text
    
    df_clean = df.copy()
    df_clean['objetivo_limpio'] = df_clean[col_obj].apply(clean_objective_text)
    
    try:
        dispersion = df_clean.groupby('objetivo_limpio').agg({
            scores_col: ['count', 'std'],
        }).round(1)
        
        dispersion.columns = ['N_actividades', 'Desvio_estandar']
        
        # Calcular IQR
        iqr_values = []
        for obj in dispersion.index:
            obj_data = df_clean[df_clean['objetivo_limpio'] == obj][scores_col]
            iqr = obj_data.quantile(0.75) - obj_data.quantile(0.25) if len(obj_data) > 1 else 0
            iqr_values.append(round(iqr, 1))
        
        dispersion['IQR'] = iqr_values
        dispersion = dispersion.sort_values('Desvio_estandar', ascending=False).reset_index()
        
        return dispersion.head(8)
    except Exception:
        return pd.DataFrame()

def find_improvement_opportunities(df: pd.DataFrame, col_obj: str, col_act: str, 
                                 scores_col: str, min_improvement: float = 15.0) -> pd.DataFrame:
    """Encuentra actividades que podrían reubicarse en otros objetivos"""
    
    if len(df) == 0 or scores_col not in df.columns:
        return pd.DataFrame()
    
    opportunities = []
    
    try:
        for idx, row in df.iterrows():
            current_activity = normalize_text(str(row[col_act]))
            current_obj = str(row.get(col_obj, "Sin objetivo"))
            current_score = float(row[scores_col])
            
            best_alternative_score = 0
            best_alternative_obj = None
            
            unique_objectives = df[col_obj].dropna().unique()
            
            for alt_obj in unique_objectives:
                if str(alt_obj) == current_obj:
                    continue
                    
                alt_obj_text = normalize_text(str(alt_obj))
                if alt_obj_text:  # Solo si el objetivo alternativo tiene contenido
                    alt_score = fuzz.token_set_ratio(current_activity, alt_obj_text)
                    
                    if alt_score > best_alternative_score:
                        best_alternative_score = alt_score
                        best_alternative_obj = alt_obj
            
            improvement = best_alternative_score - current_score
            
            if improvement >= min_improvement and best_alternative_obj:
                opportunities.append({
                    'Actividad': str(row[col_act])[:60] + "..." if len(str(row[col_act])) > 60 else str(row[col_act]),
                    'Obj_actual': current_obj[:40] + "..." if len(current_obj) > 40 else current_obj,
                    'Score_actual': round(current_score, 1),
                    'Obj_sugerido': str(best_alternative_obj)[:40] + "..." if len(str(best_alternative_obj)) > 40 else str(best_alternative_obj),
                    'Score_sugerido': round(best_alternative_score, 1),
                    'Mejora_pp': round(improvement, 1)
                })
        
        opportunities_df = pd.DataFrame(opportunities)
        if not opportunities_df.empty:
            opportunities_df = opportunities_df.sort_values('Mejora_pp', ascending=False)
        
        return opportunities_df.head(15)
    except Exception:
        return pd.DataFrame()

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
            if i in processed or not norm1:  # Skip si ya procesado o vacío
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

def analyze_missing_objectives(df: pd.DataFrame, col_obj: str, col_act: str) -> Dict[str, Any]:
    """Analiza actividades sin objetivo asignado"""
    
    missing_obj = df[df[col_obj].apply(is_empty_value) & df[col_act].apply(lambda x: is_meaningful_text(x, min_length=3))]
    
    return {
        'total_sin_objetivo': len(missing_obj),
        'porcentaje_sin_objetivo': round(len(missing_obj) / len(df) * 100, 1) if len(df) > 0 else 0,
        'ejemplos': missing_obj[col_act].head(5).tolist() if len(missing_obj) > 0 else []
    }

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
    improvements = len(report.get('potencial_mejora', []))
    if improvements > 10:
        recommendations['mediano_plazo'].append(
            f"Evaluar reubicación de actividades con alto potencial de mejora"
        )
    
    recommendations['mediano_plazo'].append(
        "Establecer trazabilidad KPI/evidencias para cada objetivo específico"
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
st.subheader("🔍 Diagnóstico de Datos")
total_meaningful = count_all_activities(df, col_act)
total_with_obj = int((~df[col_obj].apply(is_empty_value)).sum())
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

# Mostrar ejemplos de problemas
if total_meaningful > total_both:
    st.info(f"💡 Se analizarán **{total_meaningful}** actividades (incluyendo {total_meaningful - total_both} sin objetivo asignado)")
    
    with st.expander("Ver ejemplos de actividades sin objetivo asignado"):
        sin_obj_examples = df[df[col_obj].apply(is_empty_value) & df[col_act].apply(lambda x: is_meaningful_text(x, min_length=3))]
        if not sin_obj_examples.empty:
            st.dataframe(sin_obj_examples[[col_obj, col_act]].head(5), use_container_width=True)

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
if st.button("🔍 Realizar Análisis Avanzado", type="primary"):
    with st.spinner("Calculando consistencias..."):
        sum_indep, det_indep = compute_pairwise_consistency_single(df, uploaded.name, col_obj, col_act, thr)
        
        # Generar análisis avanzado
        advanced_report = generate_advanced_report(det_indep, col_obj, col_act)
        
        # Mostrar resumen ejecutivo
        st.markdown("---")
        st.markdown(format_report_for_display(advanced_report))
        
        # Mostrar tablas detalladas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Objetivos Críticos (Top 10)")
            if not advanced_report['rendimiento_objetivos'].empty:
                st.dataframe(advanced_report['rendimiento_objetivos'], use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar objetivos críticos")
        
        with col2:
            st.subheader("📈 Mayor Dispersión Interna")
            if not advanced_report['dispersion_objetivos'].empty:
                st.dataframe(advanced_report['dispersion_objetivos'], use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar dispersión")
        
        st.subheader("🔄 Actividades con Potencial de Mejora/Reubicación")
        if not advanced_report['potencial_mejora'].empty:
            st.dataframe(advanced_report['potencial_mejora'], use_container_width=True)
            st.caption("💡 **Nota**: La reubicación debe considerarse luego de reelaborar la redacción de la actividad.")
        else:
            st.info("No se encontraron actividades con alto potencial de mejora")
        
        st.subheader("🔍 Actividades Duplicadas o Similares")
        if not advanced_report['duplicados'].empty:
            st.dataframe(advanced_report['duplicados'], use_container_width=True)
            st.caption("💡 **Sugerencia**: Consolidar duplicadas como líneas de trabajo con sub-tareas medibles")
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
        st.subheader("📋 Plan de Mejora por Etapas")
        recs = advanced_report['recomendaciones']
        
        with st.expander("🟡 Corto Plazo (0-30 días)", expanded=True):
            for rec in recs['corto_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("🟠 Mediano Plazo (1-3 meses)"):
            for rec in recs['mediano_plazo']:
                st.write(f"• {rec}")
        
        with st.expander("🔴 Largo Plazo (3+ meses)"):
            for rec in recs['largo_plazo']:
                st.write(f"• {rec}")
        
        # Guía de Reescritura
        st.subheader("✍️ Guía Práctica de Reescritura")
        st.markdown("""
        **Plantilla sugerida para objetivos y actividades:**
        
        `Verbo operativo + Objeto + Ámbito/Población + Entregable + Resultado esperado`
        
        **Ejemplos:**
        - ✅ **Buena**: "Implementar tablero de seguimiento en Looker Studio para objetivos que contemple actualización mensual → Indicadores disponibles y monitoreados"
        - ✅ **Buena**: "Diseñar e institucionalizar protocolo de autoevaluación anual → Informes de autoevaluación y plan de mejora"
        - ❌ **Mala**: "Realizar actividades de capacitación"
        """)

        # Descargas
        st.subheader("💾 Descargas")
        
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar bloques para Excel
        blocks = [
            ("Resumen_Estadisticas", pd.DataFrame([advanced_report['estadisticas_generales']])),
            ("Distribucion_Niveles", pd.DataFrame([advanced_report['distribucion_niveles']])),
            ("Sin_Objetivo", pd.DataFrame([advanced_report['sin_objetivo']])),
            ("Objetivos_Criticos", advanced_report['rendimiento_objetivos']),
            ("Dispersion_Objetivos", advanced_report['dispersion_objetivos']),
            ("Potencial_Mejora", advanced_report['potencial_mejora']),
            ("Actividades_Duplicadas", advanced_report['duplicados']),
            ("Detalle_Completo", det_indep),
        ]
        
        # Generar Excel
        excel_bytes = excel_from_blocks(blocks)
        
        st.download_button(
            "⬇️ Descargar Análisis Excel Completo", 
            data=excel_bytes, 
            file_name=f"analisis_consistencia_avanzado_{ts}.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Contra el PEI (opcional)
st.subheader("🎯 Análisis contra el PEI (opcional)")
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
