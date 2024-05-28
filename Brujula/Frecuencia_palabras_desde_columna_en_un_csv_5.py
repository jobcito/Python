import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Lista de artículos e ilativos a excluir
palabras_a_quitar = set([
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 
    'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 
    'so', 'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'y', 'o', 'u', 'que', 'como', 'aunque', 'pero', 'sino', 'porque', 'pues', 
    'mientras', 'si', 'cuando', 'donde', 'nan', 'etc', 'se', 'no', 'es', 'al', 'respecto', 'del', 'le', 'consulta', 'consulto'
])

def procesar_csv(input_file, text_columns, output_csv, output_wordcloud, output_bar_chart):
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
    textos_combinados = []
    for col in text_columns:
        textos_combinados.extend(df[col].astype(str).values)
    texto = ' '.join(textos_combinados)
    
    # Eliminar caracteres no alfabéticos y convertir a minúsculas
    texto = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', '', texto).lower()
    
    # Dividir el texto en palabras
    words = texto.split()
    
    # Filtrar las palabras excluyendo las palabras_a_quitar
    palabras_filtradas = [word for word in words if word not in palabras_a_quitar]
    
    # Contar la frecuencia de cada palabra
    word_counts = Counter(palabras_filtradas)
    
    # Guardar las frecuencias en un archivo CSV
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Palabra', 'Frecuencia'])

    # Ordenar el DataFrame por la columna 'Frecuencia' en orden descendente
    word_counts_df_sorted = word_counts_df.sort_values(by='Frecuencia', ascending=False)
    
    # Guardar el DataFrame ordenado en un archivo CSV
    word_counts_df_sorted.to_csv(output_csv, index=False)
    
    # Generar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    
    # Mostrar y guardar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_wordcloud)
    plt.show()

    # Limitar a las primeras 50 palabras más frecuentes
    top_50_words = word_counts_df_sorted.head(50)
    
    # Generar el gráfico de barras
    plt.figure(figsize=(10, 5))
    bars = plt.bar(top_50_words['Palabra'], top_50_words['Frecuencia'])
    plt.xlabel('Palabra')
    plt.ylabel('Frecuencia')
    plt.title('Top 50 palabras más frecuentes')
    plt.xticks(rotation=70, fontsize=8)

    # Agregar etiquetas encima de las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=8)


    plt.tight_layout()
    plt.savefig(output_bar_chart)
    plt.show()

# Especificar los nombres de los archivos
input_file = 'Registro_brujula.csv'  # Nombre del archivo CSV de entrada
text_columns = ['PREGUNTA_1', 'PREGUNTA_2', 'PREGUNTA_3']  # Nombres de las columnas que contienen el texto
output_csv = 'frecuencia_palabras.csv'  # Nombre del archivo CSV de salida
output_wordcloud = 'nube_palabras.png'  # Nombre de la imagen de la nube de palabras
output_bar_chart = 'grafico_barras.png'  # Nombre de la imagen del gráfico de barras

# Llamar a la función para procesar el archivo CSV
procesar_csv(input_file, text_columns, output_csv, output_wordcloud, output_bar_chart)
