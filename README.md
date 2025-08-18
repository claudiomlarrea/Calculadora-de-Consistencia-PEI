# Análisis de consistencia de actividades PEI – Multi-archivo (v10.2)

La app acepta hasta 6 planillas (CSV/XLSX), limpia el objetivo (solo `1.x …`), excluye “Sin objetivo (vacío)”, calcula la **consistencia** por actividad y sugiere el **Objetivo de máxima consistencia**.  

## Salidas
- **Excel consolidado** (`Informe` e `Informe+Fuente`).
- **Word con la estructura EXACTA del ejemplo**:
  - **RESUMEN**
  - **A- Análisis de coherencia**  
    1. Panorama General (plena/parcial/desvío)  
    2. Principales hallazgos por objetivos (1..6)  
    3. Recomendaciones estratégicas  
  - **B- Grado de desarrollo del PEI por objetivo específico**  
    1) Objetivos mayormente desarrollados  
    2) Objetivos con registro insuficiente o con desvíos  
    3) Síntesis

> En la barra lateral podés **editar los nombres** de los 6 objetivos (ej.: “Aseguramiento de la calidad”) para que el informe Word se lea exactamente como el documento de referencia.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py

