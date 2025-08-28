# Reemplaza la sección de DIAGNÓSTICOS DETALLADOS en tu app.py con esto:

# DIAGNÓSTICOS DETALLADOS
st.subheader("🔍 Diagnóstico Detallado de Datos")

# Importar la nueva función
from utils import count_with_objective

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
st.write("### 📋 Análisis de Validez por Tipo")

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

# Mostrar resumen de criterios
with st.expander("🔍 Criterios de Validez Actuales"):
    st.write("""
    **Una actividad se considera VÁLIDA si:**
    - No es un valor nulo (NaN)  
    - No está en la lista de tokens vacíos: "", "nan", "none", "null", "n/a", "na", "-", "–", "—"
    - Tiene al menos 1 carácter que no sea solo espacios
    
    **Una actividad se considera INVÁLIDA si:**
    - Es un valor nulo (NaN) en pandas
    - Es exactamente uno de los tokens vacíos listados arriba
    - Es una cadena completamente vacía o solo espacios
    
    **Nota:** Actividades válidas sin objetivo asignado se procesan usando un texto genérico de comparación.
    """)

if total_meaningful != df_original_size:
    excluded_count = df_original_size - total_meaningful
    st.warning(f"⚠️ Se excluyeron {excluded_count} registros ({excluded_count/df_original_size*100:.1f}%) por no tener contenido válido en la columna de actividades.")
