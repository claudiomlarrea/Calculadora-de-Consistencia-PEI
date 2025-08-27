
# Calculadora de Consistencia PEI – UCCuyo 2023–2027

Esta calculadora permite analizar la coherencia entre las actividades institucionales registradas en el Plan Estratégico Institucional (PEI) de la UCCuyo y sus objetivos específicos para el período 2023–2027.

## 🚀 ¿Qué hace esta calculadora?
- Acepta directamente los **6 archivos CSV** de los objetivos del PEI.
- Calcula automáticamente:
  - Total de actividades por archivo.
  - Actividades con **consistencia plena**, **parcial** o **nula**.
- Genera dos archivos descargables:
  - 📊 Un Excel con tabla resumen.
  - 📄 Un documento Word narrado con el análisis.

## 📁 Archivos requeridos
Subir **6 archivos .CSV**, uno por cada objetivo específico del PEI.

## 🧑‍💻 Cómo ejecutar localmente

1. Clonar este repositorio o descargar los archivos.
2. Crear un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:

```bash
streamlit run app.py
```

## 📦 Despliegue en Streamlit Cloud
Podés subir los 3 archivos (`app.py`, `requirements.txt`, `README.md`) a un repositorio de GitHub y desplegarlo en [streamlit.io](https://streamlit.io/). No se necesita archivo adicional de referencia.

---

Desarrollado para la Secretaría de Investigación – UCCuyo.
