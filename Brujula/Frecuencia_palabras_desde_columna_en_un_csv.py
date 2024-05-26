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

def process_csv(input_file, text_column, output_csv, output_image):
    # Leer el archivo CSV
    df = pd.read_csv(input_file)
    
    # Concatenar todos los textos de la columna especificada
    text = ' '.join(df[text_column].astype(str).values)
    
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
input_file = 'Pregunta_1.csv'  # Nombre del archivo CSV de entrada
text_column = 'PREGUNTA_1'  # Nombre de la columna que contiene el texto
output_csv = 'frecuencia_palabras.csv'  # Nombre del archivo CSV de salida
output_image = 'nube_palabras.png'  # Nombre de la imagen de la nube de palabras

# Llamar a la función para procesar el archivo CSV
process_csv(input_file, text_column, output_csv, output_image)
