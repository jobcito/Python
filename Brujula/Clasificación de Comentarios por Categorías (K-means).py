import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from collections import Counter
import os

# 1. Configuración inicial
ARCHIVO_ENTRADA = 'Registro_Asistencia_Brujula_2025.xlsx'
COLUMNAS_TEXTO = ['PREGUNTA_1', 'PREGUNTA_2', 'PREGUNTA_3']
NUM_CATEGORIAS = 5
RANDOM_STATE = 42

# 2. Stopwords personalizadas
STOPWORDS = set([
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en',
    'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin',
    'so', 'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'y', 'o', 'u', 'que', 'como', 'aunque', 'pero', 'sino', 'porque', 'pues',
    'mientras', 'si', 'cuando', 'donde', 'nan', 'etc', 'se', 'no', 'es', 'al', 'respecto',
    'del', 'le', 'consulta', 'consulto', 'ser', 'haber', 'este', 'esta', 'estos', 'estas'
])

# 3. Función para guardar gráficos
def guardar_grafico(nombre_archivo, dpi=300):
    nombre_completo = f"KMeans_Comentarios_{nombre_archivo}.png"
    plt.savefig(nombre_completo, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico guardado como: {nombre_completo}")
    plt.close()

# 4. Preprocesamiento de texto
def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', ' ', texto.lower())
    palabras = [word for word in texto.split()
                if word not in STOPWORDS and len(word) > 2 and not word.isnumeric()]
    return " ".join(palabras)

# 5. Cargar y preparar datos
try:
    df = pd.read_excel(ARCHIVO_ENTRADA, engine='openpyxl')
    print(f"\nColumnas disponibles: {df.columns.tolist()}")

    columnas_existentes = [col for col in COLUMNAS_TEXTO if col in df.columns]
    if not columnas_existentes:
        raise ValueError(f"No se encontraron las columnas de texto especificadas: {COLUMNAS_TEXTO}")

    print(f"Procesando columnas: {columnas_existentes}")

    textos_combinados = []
    for col in columnas_existentes:
        textos_combinados.extend(df[col].fillna('').apply(preprocesar_texto).tolist())

    textos_combinados = [texto for texto in textos_combinados if texto.strip()]

    if not textos_combinados:
        raise ValueError("No se encontraron textos válidos para analizar")

    print(f"Número de comentarios válidos encontrados: {len(textos_combinados)}")

except Exception as e:
    print(f"\nError al cargar/processar el archivo: {str(e)}")
    exit()

# 6. Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(STOPWORDS))
X = vectorizer.fit_transform(textos_combinados)

# 7. Análisis de clusters óptimos - VERSIÓN CORREGIDA
print("\nCalculando número óptimo de categorías...")
min_clusters = 2
max_clusters = min(10, len(textos_combinados) - 1)
possible_k_values = range(min_clusters, max_clusters + 1)

inertias = []
silhouettes = []
valid_k_values = []

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

    if k > 1 and len(textos_combinados) > k:
        try:
            silhouette = silhouette_score(X, kmeans.labels_)
            silhouettes.append(silhouette)
            valid_k_values.append(k)
        except:
            pass

# Gráficos de análisis corregidos
plt.figure(figsize=(12, 5))

# Gráfico del codo
plt.subplot(1, 2, 1)
plt.plot(possible_k_values, inertias, 'bo-')
plt.xlabel('Número de categorías')
plt.ylabel('Inercia')
plt.title('Método del Codo')

# Gráfico de silhouette
if len(silhouettes) > 0:
    plt.subplot(1, 2, 2)
    plt.plot(valid_k_values, silhouettes, 'go-')
    plt.xlabel('Número de categorías')
    plt.ylabel('Puntaje Silhouette')
    plt.title('Análisis Silhouette')

plt.tight_layout()
guardar_grafico("Analisis_Optimo_Categorias")

# 8. Clustering K-means
if NUM_CATEGORIAS > max_clusters:
    NUM_CATEGORIAS = max_clusters
    print(f"\nAjustando número de categorías a {NUM_CATEGORIAS} (máximo posible)")

print(f"\nCreando {NUM_CATEGORIAS} categorías con K-means...")
kmeans = KMeans(n_clusters=NUM_CATEGORIAS, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X)

# Asignar categorías al dataframe original
for i, col in enumerate(columnas_existentes):
    mask = df[col].notna()
    textos = df.loc[mask, col].apply(preprocesar_texto)
    if not textos.empty:
        X_col = vectorizer.transform(textos)
        df.loc[mask, 'CATEGORIA_KMEANS'] = kmeans.predict(X_col) + 1

# 9. Palabras clave por categoría
print("\nExtrayendo palabras clave por categoría...")
categorias_palabras_clave = {}
for i in range(NUM_CATEGORIAS):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    palabras = ' '.join([textos_combinados[idx] for idx in cluster_indices]).split()
    categorias_palabras_clave[i+1] = [word for word, count in Counter(palabras).most_common(10)]

# 10. WordCloud por categoría
print("\nGenerando WordClouds por categoría...")
for categoria, palabras in categorias_palabras_clave.items():
    texto = ' '.join(palabras)
    if texto.strip():
        wc = WordCloud(width=800, height=400, background_color='white', collocations=False)
        wc.generate(texto)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Categoría {categoria} - Palabras Clave')
        guardar_grafico(f"WordCloud_Categoria_{categoria}")

# 11. Distribución de categorías
print("\nGenerando gráfico de distribución...")
distribucion = df['CATEGORIA_KMEANS'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
bars = plt.bar(distribucion.index.astype(str), distribucion.values, color='skyblue')
plt.title('Distribución de Comentarios por Categoría (K-means)')
plt.xlabel('Categoría')
plt.ylabel('Número de Comentarios')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.xticks(rotation=0)
plt.tight_layout()
guardar_grafico("Distribucion_Categorias")

# 12. Exportar resultados
print("\nExportando resultados...")
df.to_excel('Comentarios_Clasificados_KMeans.xlsx', index=False)

resumen_categorias = pd.DataFrame({
    'Categoría': [f'Categoría {i}' for i in categorias_palabras_clave.keys()],
    'Palabras Clave': [', '.join(words) for words in categorias_palabras_clave.values()],
    'Cantidad Comentarios': [sum(kmeans.labels_ == i) for i in range(NUM_CATEGORIAS)]
})

resumen_categorias.to_excel('Resumen_Categorias_KMeans.xlsx', index=False)

print("\nProceso completado exitosamente!")
print(f"- Archivo con comentarios clasificados: 'Comentarios_Clasificados_KMeans.xlsx'")
print(f"- Resumen de categorías: 'Resumen_Categorias_KMeans.xlsx'")
print(f"- Gráficos guardados en el directorio actual")
