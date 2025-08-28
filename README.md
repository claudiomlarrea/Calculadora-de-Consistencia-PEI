
# Calculadora Consistencia PEI UCCuyo (Streamlit)

Subí 6 CSV (objetivos 1–6) y descarga automáticamente:
- `consistencia_por_objetivo.xlsx` (detalle + resúmenes)
- `Informe_consistencia_PEI_UCCuyo.docx` (metodología, resultados, análisis y conclusiones)

## Archivos
- `app.py`
- `utils.py`
- `requirements.txt`
- `runtime.txt` *(opcional en Streamlit Cloud)*

## Despliegue
1. Subí estos archivos a un repo de GitHub.
2. En Streamlit Cloud, elegí el repo y como **Main file**: `app.py`.
3. Con `requirements.txt` alcanza; `runtime.txt` es opcional.

## Notas
- Se ignoran filas con “-” o sin actividad.
- La columna **“Consistencia del objetivo (%)”** se agrega al detalle.
- La consistencia se detecta si “Objetivos específicos X” inicia con `X.n` (p. ej., `1.5`). Si no hay código pero menciona el objetivo → **Parcial**; de lo contrario → **Sin correspondencia**.
