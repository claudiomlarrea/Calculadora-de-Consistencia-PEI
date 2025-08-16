# Análisis de consistencia de actividades PEI – **Multi-archivo** (v8)

La app permite subir **hasta 6 planillas** (CSV/XLSX), una por cada objetivo, por ejemplo:
- `Plan Estratégico Institucional UCCuyo_Objetivo 1_Tabla.csv`
- …
- `Plan Estratégico Institucional UCCuyo_Objetivo 6_Tabla.csv`

## ¿Qué hace?
- **Detecta** automáticamente las columnas de *Objetivo específico* y *Actividad*.
- **Limpia** el objetivo para quedarse **solo con el tramo `1.x …`** (evita que se mezclen actividades/resultados).
- **Excluye** filas con *“Sin objetivo (vacío)”*.
- Calcula **% de consistencia** por actividad y el **promedio global**.
- Entrega un **Excel** consolidado:
  - Hoja **Informe** (4 columnas: Objetivo, Actividad, % actividad, Promedio global).
  - Hoja **Informe+Fuente** (agrega la columna *Fuente (archivo)* para trazabilidad).
- Genera un **Word** con conclusiones (cantidad de actividades, promedio global e interpretación por niveles).

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py
