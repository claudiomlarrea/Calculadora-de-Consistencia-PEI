# Análisis de consistencia de actividades PEI

Genera un **único informe en Excel** con 4 columnas, luego de analizar la correlación/consistencia entre cada *Actividad* y el *Objetivo específico* vinculado:

1. **Objetivo específico**
2. **Actividad específica cargada**
3. **Porcentaje de correlación o consistencia de cada actividad** (0–100, 1 decimal)
4. **Porcentaje de correlación total promedio** (mismo valor para todas las filas; promedio global de la col. 3)

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py

