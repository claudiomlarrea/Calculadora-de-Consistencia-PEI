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
    excel_from_blocks, normalize_text, is_empty_value
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
    valid_activities = df[~df[col_act].apply(is_empty_value)]
    total_activities = len(valid_activities)
    scores = valid_activities[scores_col].astype(float)
    
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
    classification_counts = valid_activities[classification_col].value_counts()
    report['distribucion_niveles'] = {
        'alta_75_plus': classification_counts.get('plena', 0),
        'media_50_74': classification_counts.get('parcial', 0),
        'baja_menor_50': classification_counts.get('nula', 0)
    }
    
    # 3. Rendimiento por objetivo específico
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
    
    # 7. Recomendaciones
    recommendations = generate_recommendations(report)
    report['recomendaciones'] = recommendations
    
    return report

def analyze_objectives_performance(df: pd.DataFrame, col_obj: str, scores_col: str, 
                                 classification_col: str) -> pd.DataFrame:
    """Analiza el rendimiento por objetivo específico"""
    
    def clean_objective_text(obj_text):
        if pd.isna(obj_text):
            return obj_text
        text = str(obj_text).strip()
        match = re.match(r'^(\d+\.\d+)\s*[.-]*\s*(.*?)(?:\s*(?:Actividades|Resultados|Indicadores).*)?$', 
                        text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"{match.group(1)} {match.group(2).strip()}"
        return text
    
    df_clean = df.copy()
    df_clean['objetivo_limpio'] = df_clean[col_obj].apply(clean_objective_text)
    
    grouped = df_clean.groupby('objetivo_limpio').agg({
        scores_col: ['count', 'mean', 'median', 'std'],
        classification_col: lambda x: (x == 'nula').sum() / len(x) * 100
    }).round(1)
    
    grouped.columns = ['N_actividades', 'Promedio', 'Mediana', 'Desvio_std', 'Porc_en_Bajo']
    grouped = grouped.sort_values('Promedio').reset_index()
    
    # Calcular IQR
    grouped['IQR'] = df_clean.groupby('objetivo_limpio')[scores_col].apply(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).round(1).values
    
    return grouped.head(10)  # Top 10 críticos

def analyze_objectives_dispersion(df: pd.DataFrame, col_obj: str, scores_col: str) -> pd.DataFrame:
    """Analiza la dispersión interna de objetivos"""
    
    def clean_objective_text(obj_text):
        if pd.isna(obj_text):
            return obj_text
        text = str(obj_text).strip()
        match = re.match(r'^(\d+\.\d+)\s*[.-]*\s*(.*?)(?:\s*(?:Actividades|Resultados|Indicadores).*)?$', 
                        text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"{match.group(1)} {match.group(2).strip()}"
        return text
    
    df_clean = df.copy()
    df_clean['objetivo_limpio'] = df_clean[col_obj].apply(clean_objective_text)
    
    dispersion = df_clean.groupby('objetivo_limpio').agg({
        scores_col: ['count', 'std'],
    }).round(1)
    
    dispersion.columns = ['N_actividades', 'Desvio_estandar']
    
    # Calcular IQR
    iqr_data = df_clean.groupby('objetivo_limpio')[scores_col].apply(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).round(1)
    
    dispersion['IQR'] = iqr_data
    dispersion = dispersion.sort_values('Desvio_estandar', ascending=False).reset_index()
    
    return dispersion.head(8)

def find_improvement_opportunities(df: pd.DataFrame, col_obj: str, col_act: str, 
                                 scores_col: str, min_improvement: float = 15.0) -> pd.DataFrame:
    """Encuentra actividades que podrían reubicarse en otros objetivos"""
    
    opportunities = []
    
    for idx, row in df.iterrows():
        current_activity = normalize_text(str(row[col_act]))
        current_obj = str(row[col_obj])
        current_score = float(row[scores_col])
        
        best_alternative_score = 0
        best_alternative_obj = None
        
        unique_objectives = df[col_obj].dropna().unique()
        
        for alt_obj in unique_objectives:
            if alt_obj == current_obj:
                continue
                
            alt_obj_text = normalize_text(str(alt_obj))
            alt_score = fuzz.token_set_ratio(current_activity, alt_obj_text)
            
            if alt_score > best_alternative_score:
                best_alternative_score = alt_score
                best_alternative_obj = alt_obj
        
        improvement = best_alternative_score - current_score
        
        if improvement >= min_improvement:
            opportunities.append({
                'Actividad': str(row[col_act])[:80] + "..." if len(str(row[col_act])) > 80 else str(row[col_act]),
                'Obj_actual': current_obj[:50] + "..." if len(current_obj) > 50 else current_obj,
                'Score_actual': round(current_score, 1),
                'Obj_sugerido': best_alternative_obj[:50] + "..." if len(str(best_alternative_obj)) > 50 else str(best_alternative_obj),
                'Score_sugerido': round(best_alternative_score, 1),
                'Mejora_pp': round(improvement, 1)
            })
    
    opportunities_df = pd.DataFrame(opportunities)
    if not opportunities_df.empty:
        opportunities_df = opportunities_df.sort_values('Mejora_pp', ascending=False)
    
    return opportunities_df.head(15)

def find_duplicate_activities(df: pd.DataFrame, col_act: str, threshold: float = 85.0) -> pd.DataFrame:
    """Encuentra actividades duplicadas o muy similares"""
    
    activities = df[col_act].dropna().astype(str).tolist()
    normalized_activities = [normalize_text(act) for act in activities]
    
    duplicates_info = defaultdict(list)
    processed = set()
    
    for i, (act1, norm1) in enumerate(zip(activities, normalized_activities)):
        if i in processed:
            continue
            
        similar_group = [act1]
        processed.add(i)
        
        for j, (act2, norm2) in enumerate(zip(activities, normalized_activities)):
            if i >= j or j in processed:
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
            "Implementar higiene de redacción con plantillas estandarizadas para actividades"
        )
        recommendations['corto_plazo'].append(
            "Crear glosario de verbos operativos y entregables específicos"
        )
    
    if not duplicates.empty:
        recommendations['corto_plazo'].append(
            f"Consolidar {len(duplicates)} grupos de actividades duplicadas identificadas"
        )
    
    # Recomendaciones de mediano plazo
    low_performance_objs = len(report.get('rendimiento_objetivos', []))
    if low_performance_objs > 5:
        recommendations['mediano_plazo'].append(
            f"Reencuadrar objetivos específicos con baja consistencia"
        )
    
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

total_valid = count_valid_pairs(df, col_obj, col_act)
st.info(f"**Total de actividades (objetivo & actividad no vacíos): {total_valid}**")

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
