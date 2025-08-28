
# Calculadora de Consistencia – Formulario Único

Genera un Excel con **resumen, porcentajes, rendimiento por objetivo, duplicadas, detalle** y **sugerencias** por actividad:
- **Mejor objetivo propuesto**
- **Actividad (seleccionada)**
- **% si se ubica en objetivo propuesto**
- **Δ p.p.** respecto al objetivo actual

## Uso (Streamlit Cloud)
1. En GitHub (raíz): `app.py`, `utils.py`, `requirements.txt`, `runtime.txt` (3.11.9).
2. En Streamlit Cloud:
   - Main file path: `app.py`
   - Python: 3.11 (deja `runtime.txt`)
   - Restart y luego Clear cache

## En la app
1. Subí el **Formulario Único** (XLSX/CSV).
2. Elegí columnas **Objetivo específico** y **Actividad**.
3. Ajustá umbrales y corré el análisis.
4. Descargá el Excel.
