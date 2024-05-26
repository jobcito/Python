# Importando las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargando el conjunto de datos
iris = load_iris()
X = iris.data
y = iris.target

# Dividiendo el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creando el modelo de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenando el modelo
clf.fit(X_train, y_train)

# Haciendo predicciones
y_pred = clf.predict(X_test)

# Calculando la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'La precisión del modelo de árbol de decisión es: {accuracy}')
