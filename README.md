# Análisis de consistencia de actividades PEI

Calculadora en Streamlit para estimar la relación entre actividades y objetivos específicos del PEI y generar resultados en **Excel** y un **informe en Word**.

## Funcionalidades
- Subir el Excel exportado del Formulario Único para el PEI.
- Detecta automáticamente columnas clave (objetivo, actividad, unidad, responsable).
- Calcula un porcentaje de consistencia combinando similitud de texto y solapamiento de palabras clave.
- Clasifica en Alta, Media o Baja consistencia.
- Genera:
  - Tabla por actividad
  - Resumen por objetivo
  - Resumen por unidad académica
- Exporta:
  - Resultados completos a Excel
  - Informe resumido en Word con conclusiones automáticas

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud
1. Subir a GitHub
2. Crear app en Streamlit Cloud apuntando a `app.py`
3. Usar interfaz para subir Excel y descargar informes
