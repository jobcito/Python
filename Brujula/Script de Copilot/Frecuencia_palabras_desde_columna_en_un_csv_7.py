import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Carga el archivo csv
df = pd.read_csv('Registro_brujula.csv', sep=';')

# Reemplaza los valores NaN o None con una cadena de texto vacía
df.fillna('', inplace=True)

# Procesa las tres columnas
df['texto'] = df['PREGUNTA_1'] + ' ' + df['PREGUNTA_2'] + ' ' + df['PREGUNTA_3']

# Convierte todo el texto a minúsculas
df['texto'] = df['texto'].str.lower()

# Separa el texto en palabras
df['palabras'] = df['texto'].apply(nltk.word_tokenize)

# Quita los ilativos, artículos y caracteres no alfabéticos
stop_words = set(stopwords.words('spanish'))

# Agrega las palabras que quieres excluir
palabras_excluidas = set(['consulta'])
stop_words = stop_words.union(palabras_excluidas)

df['palabras'] = df['palabras'].apply(lambda palabras: [palabra for palabra in palabras if palabra.isalpha() and palabra not in stop_words])

# Concatena todas las palabras en una lista
palabras = [palabra for sublist in df['palabras'].tolist() for palabra in sublist]

# Calcula las frecuencias de cada palabra
frecuencias = nltk.FreqDist(palabras)

# Genera un wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frecuencias)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Crea un archivo .csv de salida
df_salida = pd.DataFrame(list(frecuencias.items()), columns=['palabra', 'frecuencia'])
df_salida.sort_values('frecuencia', ascending=False, inplace=True)
df_salida.to_csv('salida.csv', index=False)

# Crea un gráfico de barras
plt.figure(figsize=(10, 5))
barplot = sns.barplot(x='palabra', y='frecuencia', data=df_salida.head(50))
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')
plt.title('Top 50 palabras más frecuentes')
plt.xticks(rotation=70)

# Añade el valor de las frecuencias a cada barra
for p in barplot.patches:
    height = p.get_height()
    barplot.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 

plt.show()
