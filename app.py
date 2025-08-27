import pandas as pd
from io import BytesIO
from docx import Document

def analizar_archivos(uploaded_files):
    dataframes = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        df['Archivo'] = file.name
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)

    # Consistencia simulada: columnas esperadas
    df_all['Consistencia'] = df_all.apply(lambda row: evaluar_consistencia(row), axis=1)
    return df_all

def evaluar_consistencia(row):
    if "objetivo" in row['Acción'].lower():
        return "Plena"
    elif "acciones" in row['Acción'].lower():
        return "Parcial"
    else:
        return "Baja"

def generar_excel(df):
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

def generar_word(df):
    doc = Document()
    doc.add_heading('Informe de análisis de consistencia PEI', 0)

    total = len(df)
    plena = (df['Consistencia'] == 'Plena').sum()
    parcial = (df['Consistencia'] == 'Parcial').sum()
    baja = (df['Consistencia'] == 'Baja').sum()

    doc.add_paragraph(f"Se analizaron {total} acciones.")
    doc.add_paragraph(f"{plena} con consistencia plena ({round(plena/total*100,2)}%)")
    doc.add_paragraph(f"{parcial} con consistencia parcial ({round(parcial/total*100,2)}%)")
    doc.add_paragraph(f"{baja} con baja consistencia ({round(baja/total*100,2)}%)")

    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output
