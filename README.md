# 📊 Calculadora de Consistencia PEI

Calculadora interactiva para analizar la coherencia entre las acciones cargadas en los formularios institucionales y los objetivos estratégicos del PEI de UCCuyo.

## Cómo usar

1. Subí los 6 archivos `.csv` correspondientes a los objetivos.
2. La app analizará cada acción y clasificará su consistencia como:
   - Plena
   - Parcial
   - Baja
3. Podés descargar:
   - Un archivo Excel con los resultados
   - Un informe Word con una narración del análisis

## Requisitos

- Streamlit
- Pandas
- Openpyxl
- Python-docx

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py

