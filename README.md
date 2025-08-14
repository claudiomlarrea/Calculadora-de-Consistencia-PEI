
# Análisis de consistencia de actividades PEI (v4)

Calculadora en Streamlit para estimar la relación entre actividades y objetivos específicos del PEI y generar resultados en **Excel** y un **informe en Word**.

## Novedades clave
- Selector de hoja para Excels con varias pestañas.
- Combinar "Código de objetivo" + "Objetivo (texto)".
- **Forward-fill** y **Backfill** con **agrupación por columnas** (p. ej., Unidad Académica).
- Diagnóstico de vacíos y frecuencias de objetivos.
- Umbral ajustable para marcar **inconsistentes** y **sección en el Word** con listado de actividades inconsistentes y **objetivo sugerido** por similitud.
- Promedio general **excluye** “Sin objetivo (vacío)”.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py
```

Luego, sube el Excel del Formulario PEI, ajusta columnas y opciones (ffill/backfill, agrupación, umbral) y descarga el Excel + Word.
