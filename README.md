
# Análisis de consistencia de actividades PEI

Calculadora en Streamlit para estimar la relación entre actividades y objetivos específicos del PEI y generar resultados en **Excel** y un **informe en Word**.

## Novedad
- Las filas con **Objetivo específico vacío** se marcan como **"Sin objetivo (vacío)"** y reciben **0%**.
- El **promedio general mostrado y en el Word excluye** esas filas para no sesgar el resultado.
- En tablas y Excel quedan registradas para auditoría.

## Funcionalidades
- Subir el Excel exportado del Formulario Único para el PEI.
- Detección automática de columnas clave (objetivo, actividad, unidad, responsable).
- Cálculo de porcentaje de consistencia (RapidFuzz + Jaccard).
- Clasificación: Alta (≥75), Media (50–74), Baja (<50).
- Resúmenes por objetivo y por unidad.
- Exportación a Excel y a Word con conclusiones automáticas.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud
1. Subir a GitHub
2. Crear app en Streamlit Cloud apuntando a `app.py`
3. Usar interfaz para subir Excel y descargar informes
