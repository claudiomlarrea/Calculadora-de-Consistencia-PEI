# Análisis de consistencia de actividades PEI

Mejoras para que la plantilla quede **limpia**:
- Columna **Actividad alternativa (fallback)** por si la principal está vacía.
- **Forward/Backfill para Objetivo y para Actividad**, agrupando por una columna estable (p. ej., *Unidad Académica*).
- Opción para **eliminar filas con Actividad vacía** (ON) y **quitar duplicados** por *(Objetivo, Actividad)* (ON).

**Salida:** una sola hoja `Informe` con 4 columnas:
1. Objetivo específico  
2. Actividad específica cargada  
3. Porcentaje de correlación o consistencia de cada actividad  
4. Porcentaje de correlación total promedio (mismo valor en todas las filas)

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py

