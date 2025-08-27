
# Calculadora de Consistencia PEI – UCCuyo 2023–2027

Esta aplicación en Streamlit permite evaluar la consistencia de las actividades institucionales registradas en el Formulario Único con respecto al Plan Estratégico Institucional (PEI) de la Universidad Católica de Cuyo (2023–2027).

## Características

- Verificación automática por objetivos generales y específicos del PEI.
- Validación basada en coincidencia de palabras clave y metadatos.
- Detección de desvíos y actividades sin correspondencia.
- Análisis por unidad académica, tipo de actividad y nivel de alineación.
- Reportes descargables en Excel y Word.

## Archivos principales

- `app.py`: aplicación principal de Streamlit.
- `pei_referencia.csv`: base de referencia con objetivos, acciones e indicadores del PEI.
- `requirements.txt`: dependencias necesarias para ejecutar la app.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Autor

Claudio M. Larrea Arnau
