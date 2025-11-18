import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar los datos
# Asegúrate de que el archivo se llame 'iris.csv' en la misma carpeta.
df = pd.read_csv("iris.csv", encoding='latin-1')

# 2. Preparar los datos (Separar características y etiquetas)
# X (Características/Features): Todas las filas, columnas 0 a 3 (las 4 medidas de la flor)
X = df.iloc[:, 0:4]
# y (Etiqueta/Target): La columna 'Clase' (el tipo de flor)
y = df["Clase"]

# 3. Dividir los datos para entrenamiento y prueba
# 80% para entrenar (X_train, y_train)
# 20% para probar (X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # Asegura que la división sea la misma cada vez
)

# 4. Crear y entrenar el modelo (k-NN)
# Se crea el clasificador con k=3 vecinos
knn = KNeighborsClassifier(n_neighbors=3)
# Se entrena el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)

# 5. Realizar predicciones
# El modelo predice las etiquetas para el conjunto de prueba
y_pred = knn.predict(X_test)

# -------------------------------------------------------------
# 📊 Visualización de los resultados
# -------------------------------------------------------------
# Gráfico de dispersión para comparar valores reales (y_test) vs. predichos (y_pred)
plt.scatter(y_test, y_pred, c="blue", label="Predicciones")
plt.xlabel("Flores Reales")
plt.ylabel("Flores Predichas")
plt.title("Clasificación de Flores con KNN")
plt.legend()
plt.show() # Muestra la ventana del gráfico

# 6. Evaluar el modelo y mostrar el resultado
# Se calcula la precisión comparando las predicciones con las etiquetas reales
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo k-NN: {precision * 100:.2f}%")