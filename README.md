# Análisis de consistencia de actividades PEI – Multi-archivo (v10)

Acepta hasta 6 planillas (CSV/XLSX), consolida y **limpia el objetivo** (solo `1.x …`), **excluye** “Sin objetivo (vacío)”, calcula:

- **Porcentaje de consistencia** por actividad.
- **Objetivo sugerido (máxima consistencia)** por actividad, con % sugerido y **delta (p.p.)**.

## Salidas
**Excel**
- `Informe`: Objetivo específico, Actividad, Porcentaje de consistencia, Objetivo sugerido (máxima consistencia),
  Porcentaje de consistencia (sugerido), Diferencia (p.p.), y **Promedio global**.
- `Informe+Fuente`: lo mismo + **Fuente (archivo)** para trazabilidad.

**Word (formato plantilla PEI)**
- **RESUMEN** y alcance del análisis.
- **A- Análisis de coherencia**: panorama general (plena/parcial/desvío), tabla de **hallazgos por objetivo** y **recomendaciones estratégicas**.
- **B- Grado de desarrollo del PEI por objetivo específico**: tablas de objetivos **mayormente desarrollados** y **con registro insuficiente o con desvíos**, y una **síntesis** final.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py

