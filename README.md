
# Calculadora Consistencia PEI UCCuyo (Streamlit)

**Arquitectura anti-errores de importación**
- Módulo auxiliar renombrado a `pei_utils.py` para evitar choques con paquetes llamados `utils`.
- Import de `python-docx` movido *dentro* de la función que genera el Word.

## Archivos
- `app.py` (importa desde `pei_utils.py`)
- `pei_utils.py`
- `requirements.txt`
- `runtime.txt` *(opcional en Streamlit Cloud)*

## Despliegue
1. Subí estos archivos a un repo de GitHub.
2. En Streamlit Cloud, usa **Main file**: `app.py`.
3. Al iniciar, subí **exactamente 6 CSV** (objetivos 1–6).

## Salidas automáticas
- `consistencia_por_objetivo.xlsx` (detalle + resúmenes)
- `Informe_consistencia_PEI_UCCuyo.docx` (metodología, resultados, análisis y conclusiones)
