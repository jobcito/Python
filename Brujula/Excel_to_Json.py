import pandas as pd

# Leer el archivo Excel
df = pd.read_excel('Extracción_de_preguntas_por_area_brujula.xlsx', engine='openpyxl')

# Convertir el DataFrame a JSON
json_data = df.to_json(orient='records', force_ascii=False)  # Evitar forzar a ASCII para permitir caracteres UTF-8

# Guardar el JSON en un archivo
with open('Extracción_de_preguntas_por_area_brujula.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

