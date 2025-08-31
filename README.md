
# Calculadora de consistencia PEI – Propuestas de objetivo (versión mejorada)

Esta app de Streamlit asigna **actividades** al **objetivo PEI más consistente** combinando señales semánticas y reglas de balance. 

## Novedades
- **TF‑IDF + n-gramas** sobre nombres de objetivos y **perfiles** (texto de actividades ya asociadas).
- **Solapamiento léxico** (palabras clave que detonan la coincidencia).
- **Penalización por sobre-asignación** para evitar concentrar actividades en pocos objetivos.
- **Confianza (%)** basada en la diferencia relativa **Top‑1 vs Top‑2**.
- Exporta un Excel con 4 hojas: `Propuestas`, `Discrepancias`, `Resumen`, `Objetivos_catalogo`.

## Estructura de archivos
- `app.py`: aplicación Streamlit (listo para correr).
- `requirements.txt`: dependencias.

## Cómo correr
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos de entrada (Excel)
La app detecta automáticamente las columnas (insensibles a mayúsculas/minúsculas):

- **Actividad** (o `Descripción`): texto libre de la actividad.
- **Obj. actual** *(opcional)*: nombre del objetivo actualmente asignado.
- **Obj. sugerido** *(opcional)*: sugerencia previa de la calculadora anterior.

**Opcional:** podés subir un **Catálogo de objetivos** (CSV/XLSX) con las columnas `Objetivo` y `Descripción` para mejorar la semántica.

## Parámetros (sidebar)
- Pesos de señales: nombre del objetivo, perfil de actividades actuales, perfil de sugeridas previas, solapamiento.
- `λ` de penalización por sobre-asignación.
- Umbrales de **confianza** para clasificar en `Auto`, `Revisar`, `Validación`.

## Lógica de puntaje
```text
Puntaje = 0.35*sim(nombre) + 0.30*sim(perfil_actual) + 0.25*sim(perfil_sugerido) + 0.10*overlap_lexico − λ*penalización_balance
```
Los pesos y `λ` son ajustables desde la UI.

## Salidas
- **Propuestas**: Top‑1, Top‑2, **Confianza (%)**, `Consistencia_estimada_%`, palabras clave detonantes y etiqueta de decisión (`Auto` / `Revisar` / `Validación`).
- **Discrepancias**: casos donde la nueva propuesta **difiere** del “Obj. sugerido” previo.
- **Resumen**: distribución por objetivo (**actual vs. sugerido previo vs. sugerido mejorado**).

## Notas
- Si `scikit-learn` no estuviese disponible, la app cae a un método **Jaccard** simplificado.
- Recomendación operativa: revisar primero **Discrepancias** y casos con **Confianza < 40%**.
