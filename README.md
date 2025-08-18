# Análisis de consistencia de actividades PEI – v10.3

Genera **Excel consolidado** y un **Word con la misma estructura y frases base** del informe de referencia:

- **RESUMEN**
- **A- Análisis de coherencia**  
  1. Panorama General (plena/parcial/desvío)  
  2. Principales hallazgos por objetivos (1..6) – con las frases del ejemplo  
  3. Recomendaciones estratégicas  
- **B- Grado de desarrollo del PEI por objetivo específico**  
  1) Objetivos mayormente desarrollados  
  2) Objetivos con registro insuficiente o con desvíos  
  3) Síntesis

### Umbrales configurables
En la barra lateral podés ajustar:
- **Plena correspondencia ≥** (por defecto 75)
- **Parcial ≥** (por defecto 50; lo que quede por debajo es **Desvío**)

Ajustalos para que los conteos se acerquen a tus valores (por ej. 49/41/111).

### Uso
```bash
pip install -r requirements.txt
streamlit run app.py

