# Análisis de consistencia de actividades PEI – **Multi-archivo** (v8.1)

Acepta hasta 6 planillas (CSV/XLSX), consolida y limpia el objetivo (solo `1.x …`), excluye objetivos vacíos, calcula:

- **Porcentaje de consistencia** por actividad.
- **Objetivo sugerido (máxima consistencia)** para cada actividad.
- **Porcentaje de consistencia (sugerido)** y **Diferencia (p.p.)** respecto al objetivo actual.

## Salida
**Excel** con dos hojas:
- `Informe`: Objetivo específico, Actividad, **Porcentaje de consistencia**, Objetivo sugerido, **Porcentaje de consistencia (sugerido)**, Diferencia (p.p.), **Porcentaje de consistencia total promedio**.
- `Informe+Fuente`: lo mismo + columna **Fuente (archivo)** para trazabilidad.

**Word**: conclusiones con cantidad de actividades, promedio global e interpretación por niveles.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py
