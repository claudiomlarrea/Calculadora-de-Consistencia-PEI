# Calculadora de Consistencia PEI – UCCuyo 2023–2027 (con correlación por PDF)

Esta versión cruza las **actividades** con el **PEI oficial** (PDF) y clasifica cada actividad como **Plena / Parcial / Nula** de forma **conservadora**.

## ¿Cómo funciona?
1. **Subí el PDF del PEI** (versión oficial).  
2. **Subí los 6 archivos de actividades** (CSV/XLSX/XLS). Podés subirlos en tandas; la app acumula hasta llegar a 6.  
3. Ajustá los **umbrales** (por defecto Plena=88, Parcial=68).  
4. Descargá:
   - 📊 **Excel** con Resumen (%), **Matriz por Objetivo** y **Detalle** por archivo.
   - 📄 **Word** narrado (si `python-docx` está instalado).

## Instalación local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud
Subí a GitHub, en la **raíz** del repo: `app.py`, `utils.py`, `requirements.txt`, `runtime.txt` y (opcional) `README.md`.  
En **App settings → Python version**, elegí **3.11** (o dejá `runtime.txt` con `3.11.9`).

---

**Criterio de clasificación:** similitud de texto (RapidFuzz) entre cada **actividad** y las **acciones/indicadores** del PEI; “plena” exige puntaje alto **y** coherencia de **objetivo** cuando el nombre del archivo lo insinúa (p. ej., `1_…csv`).
