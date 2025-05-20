import pandas as pd
import re
from collections import Counter
from nltk import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Cargar el archivo Excel
try:
    df = pd.read_excel('Registro_Asistencia_Brujula_2025.xlsx', engine='openpyxl')
    print("Columnas disponibles:", df.columns.tolist())
except Exception as e:
    print(f"Error al cargar el archivo: {str(e)}")
    exit()

# 2. Configuración (personalizable)
COLUMNAS_TEXTO = ['PREGUNTA_1', 'PREGUNTA_2', 'PREGUNTA_3']  # Ajustar a tus columnas reales
TOP_N = 20  # Número de bigramas/trigramas a mostrar

# 3. Lista de palabras a excluir (ampliable)
STOPWORDS = set([
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 
    'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 
    'so', 'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'y', 'o', 'u', 'que', 'como', 'aunque', 'pero', 'sino', 'porque', 'pues', 
    'mientras', 'si', 'cuando', 'donde', 'nan', 'etc', 'se', 'no', 'es', 'al', 'respecto', 
    'del', 'le', 'consulta', 'consulto', 'ser', 'haber', 'este', 'esta', 'estos', 'estas'
])

# 4. Función de preprocesamiento mejorada
def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return []
    
    # Limpieza básica
    texto = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', ' ', texto.lower())
    
    # Tokenización y filtrado
    palabras = [word for word in texto.split() 
               if word not in STOPWORDS 
               and len(word) > 2  # Eliminar palabras muy cortas
               and not word.isnumeric()]
    
    return palabras

# 5. Procesar todas las columnas de texto
palabras_combinadas = []
for col in COLUMNAS_TEXTO:
    if col in df.columns:
        df[col] = df[col].fillna('')
        palabras_combinadas.extend(df[col].apply(preprocesar_texto).sum())
    else:
        print(f"Advertencia: Columna '{col}' no encontrada")

if not palabras_combinadas:
    raise ValueError("No se encontraron datos de texto para analizar")

# 6. Generación de n-gramas
def obtener_ngramas_frecuentes(tokens, n=2, top_n=TOP_N):
    n_gramas = list(ngrams(tokens, n))
    frecuencias = Counter(n_gramas)
    return frecuencias.most_common(top_n)

# Bigramas (pares de palabras)
bigramas = obtener_ngramas_frecuentes(palabras_combinadas, n=2)
print("\nTop 20 Bigramas:")
for bigrama, freq in bigramas:
    print(f"{' '.join(bigrama)}: {freq}")

# Trigramas (ternas de palabras)
trigramas = obtener_ngramas_frecuentes(palabras_combinadas, n=3)
print("\nTop 20 Trigramas:")
for trigrama, freq in trigramas:
    print(f"{' '.join(trigrama)}: {freq}")

# 7. Visualización de resultados
def generar_wordcloud_ngramas(ngramas, title):
    ngramas_texto = [' '.join(ngrama) for ngrama, freq in ngramas for _ in range(freq)]
    texto = ' '.join(ngramas_texto)
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         collocations=False).generate(texto)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# WordCloud para bigramas
generar_wordcloud_ngramas(bigramas, "Bigramas más frecuentes")

# Gráfico de barras para trigramas
if trigramas:
    trigramas_df = pd.DataFrame(trigramas, columns=['Trigram', 'Frecuencia'])
    trigramas_df['Trigram'] = trigramas_df['Trigram'].apply(lambda x: ' '.join(x))
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(trigramas_df['Trigram'], trigramas_df['Frecuencia'], color='skyblue')
    plt.xlabel('Frecuencia')
    plt.title('Top 20 Trigramas')
    plt.gca().invert_yaxis()  # Mostrar el más frecuente arriba
    
    # Añadir etiquetas de frecuencia
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', 
                 ha='left', va='center')
    
    plt.tight_layout()
    plt.show()

# 8. Exportar resultados a Excel
resultados = {
    'Bigramas': [' '.join(b[0]) for b in bigramas],
    'Frecuencia Bigramas': [b[1] for b in bigramas],
    'Trigramas': [' '.join(t[0]) for t in trigramas],
    'Frecuencia Trigramas': [t[1] for t in trigramas]
}

pd.DataFrame(resultados).to_excel('resultados_ngramas.xlsx', index=False)
print("\nResultados exportados a 'resultados_ngramas.xlsx'")