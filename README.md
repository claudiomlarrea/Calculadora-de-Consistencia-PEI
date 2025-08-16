# Análisis de consistencia de actividades PEI – Multi-archivo (v9)

Acepta hasta 6 planillas (CSV/XLSX), consolida y limpia el objetivo (solo `1.x …`), excluye “Sin objetivo (vacío)”, calcula:

- **Porcentaje de consistencia** por actividad.
- **Objetivo sugerido (máxima consistencia)** por actividad, con % sugerido y **delta (p.p.)**.

## Salidas
**Excel**
- `Informe`: Objetivo específico, Actividad, Porcentaje de consistencia, Objetivo sugerido (máxima consistencia),
  Porcentaje de consistencia (sugerido), Diferencia (p.p.), y el **Promedio global**.
- `Informe+Fuente`: lo mismo + columna **Fuente (archivo)** para trazabilidad.

**Word avanzado**
- Resumen ejecutivo (total, promedio, mediana, P25/P75, min/máx).
- Distribución por niveles (Alta/Media/Baja).
- **Ranking por Objetivo específico** (n, promedio, mediana, % en Bajo) – top 10 críticos.
- Objetivos con **mayor dispersión** (std, IQR).
- **Actividades con alto potencial** (baja consistencia y gran mejora sugerida).
- Duplicadas/similares (detección simple).
- Guía de reescritura y **plan de mejora** (corto/mediano plazo y revisión periódica).
- Anexo metodológico (cómo se calcula la consistencia).

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py
