# Calculadora de Consistencia PEI – UCCuyo 2023–2027 (con correlación por PDF)

Cruza las **actividades** con el **PEI oficial** (PDF) y clasifica cada actividad como **Plena / Parcial / Nula** de forma **conservadora**.

## Uso
1. Subí el **PDF del PEI**.  
2. Subí los **6 archivos** (CSV/XLSX/XLS) —podés hacerlo en tandas.  
3. Ajustá **umbrales**.  
4. Descargá **Excel** (Resumen, Porcentajes, Matriz, Detalle) y **Word**.

## Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cloud
En GitHub (raíz): `app.py`, `utils.py`, `requirements.txt`, `runtime.txt` (3.11.9).  
En Streamlit Cloud seleccioná **Python 3.11** o dejá `runtime.txt`.
