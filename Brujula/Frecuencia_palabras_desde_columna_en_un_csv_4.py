import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Lista de artículos e ilativos a excluir
stop_words = set([
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 
    'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 
    'so', 'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'y', 'o', 'u', 'que', 'como', 'aunque', 'pero', 'sino', 'porque', 'pues', 
    'mientras', 'si', 'cuando', 'donde'
])

def process_csv(input_file, text_columns, output_csv, output_image):
    try:
        # Intentar leer el archivo CSV con delimitador por defecto (coma)
        df = pd.read_csv(input_file)
    except pd.errors.ParserError:
        try:
            # Si falla, intentar leer el archivo CSV con delimitador punto y coma
            df = pd.read_csv(input_file, sep=';')
        except pd.errors.ParserError:
            raise ValueError("Error al leer el archivo CSV. Verifique el delimitador y el formato del archivo.")
    
    # Concatenar todos los textos de las columnas especificadas
    text = ' '.join(df[col].astype(str).values for col in text_columns)
    
    # Eliminar caracteres no alfabéticos y convertir a minúsculas
    text = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', '', text).lower()
    
    # Dividir el texto en palabras
    words = text.split()
    
    # Filtrar las palabras excluyendo las stop_words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Contar la frecuencia de cada palabra
    word_counts = Counter(filtered_words)
    
    # Guardar las frecuencias en un archivo CSV
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Palabra', 'Frecuencia'])
    word_counts_df.to_csv(output_csv, index=False)
    
    # Generar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    
    # Mostrar y guardar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image)
    plt.show()

# Especificar los nombres de los archivos
input_file = 'Registro_brujula.csv'  # Nombre del archivo CSV de entrada
text_columns = ['PREGUNTA_1', 'PREGUNTA_2', 'PREGUNTA_3']  # Nombres de las columnas que contienen el texto
output_csv = 'frecuencia_palabras.csv'  # Nombre del archivo CSV de salida
output_image = 'nube_palabras.png'  # Nombre de la imagen de la nube de palabras

# Llamar a la función para procesar el archivo CSV
process_csv(input_file, text_columns, output_csv, output_image)
