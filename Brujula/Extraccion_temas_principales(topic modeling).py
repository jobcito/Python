import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Configuración inicial (usa las mismas que en tu código original)
palabras_a_quitar = set([
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 
    'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 
    'so', 'sobre', 'tras', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'y', 'o', 'u', 'que', 'como', 'aunque', 'pero', 'sino', 'porque', 'pues', 
    'mientras', 'si', 'cuando', 'donde', 'nan', 'etc', 'se', 'no', 'es', 'al', 'respecto', 
    'del', 'le', 'consulta', 'consulto'
])

normalizacion_palabras = {
    'majors': 'major'
}

def preprocesar_texto(texto):
    """Función para limpiar y normalizar el texto"""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', '', texto).lower()
    palabras = texto.split()
    palabras = [normalizacion_palabras.get(p, p) for p in palabras]
    palabras = [p for p in palabras if p not in palabras_a_quitar and len(p) > 2]
    return " ".join(palabras)

# 1. Cargar el archivo Excel
df = pd.read_excel('Registro_Asistencia_Brujula_2025.xlsx', engine='openpyxl')

# 2. Combinar y preprocesar todas las columnas de texto
textos_combinados = (
    df['PREGUNTA_1'].apply(preprocesar_texto) + " " + 
    df['PREGUNTA_2'].apply(preprocesar_texto) + " " + 
    df['PREGUNTA_3'].apply(preprocesar_texto)
)

# 3. Crear matriz de términos (Bag of Words)
vectorizer = CountVectorizer(
    max_df=0.95,    # Ignorar palabras muy frecuentes (en 95% de docs)
    min_df=2,       # Solo palabras que aparecen al menos en 2 documentos
    stop_words=list(palabras_a_quitar)
)
X = vectorizer.fit_transform(textos_combinados)

# 4. Aplicar LDA (Latent Dirichlet Allocation)
num_temas = 5  # Puedes ajustar este número
lda = LatentDirichletAllocation(
    n_components=num_temas,
    max_iter=10,
    learning_method='online',
    random_state=42
)
lda.fit(X)

# 5. Visualizar los temas
def imprimir_temas(modelo, vectorizer, n_palabras=10):
    palabras = vectorizer.get_feature_names_out()
    for idx, tema in enumerate(modelo.components_):
        print(f"Tema #{idx + 1}:")
        print(" ".join([palabras[i] for i in tema.argsort()[:-n_palabras - 1:-1]]))
        print()

imprimir_temas(lda, vectorizer)

# 6. Asignar el tema principal a cada comentario
temas_documentos = lda.transform(X)
df['TEMA_PRINCIPAL'] = temas_documentos.argmax(axis=1) + 1

# 7. Visualización adicional (WordCloud por tema)
for tema_num in range(num_temas):
    # Obtener palabras del tema con sus pesos
    palabras_tema = dict(zip(
        vectorizer.get_feature_names_out(),
        lda.components_[tema_num]
    ))
    
    # Generar WordCloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(palabras_tema)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Palabras Clave - Tema {tema_num + 1}')
    plt.show()

# 8. Guardar resultados
df.to_excel('comentarios_con_temas.xlsx', index=False)
