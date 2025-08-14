
# Análisis de consistencia de actividades PEI UCCuyo


# Análisis de consistencia de actividades PEI

Calculadora en Streamlit para estimar la relación entre actividades y objetivos específicos del PEI y generar resultados en **Excel** y un **informe en Word**.

## Novedades
- **Backfill** (relleno hacia arriba) además de **forward-fill** (hacia abajo), ambos con opción de **agrupar por columnas** (p.ej., Unidad Académica).
- **Selector de hoja** para Excels con varias pestañas.
- Posibilidad de **combinar "Código de objetivo" + "Objetivo (texto)"** en una sola columna.
- **Diagnóstico** con frecuencias de objetivos y conteo de vacíos.
- **Umbral ajustable** para marcar actividades **inconsistentes**.
- El **informe Word** ahora incluye una **sección al final** que enumera todas las actividades inconsistentes, indicando:
  - el **objetivo al que quedaron vinculadas**,
  - el **objetivo sugerido** según el análisis de similitud,
  - y la **similitud sugerida (%)**.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud
1. Subir a GitHub
2. Crear app en Streamlit Cloud apuntando a `app.py`
3. Usar interfaz para subir Excel, ajustar opciones y descargar informes
