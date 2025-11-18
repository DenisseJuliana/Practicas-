import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 1. Preparación de Datos
# X: Características [Tamaño (cm), Color (0=verde, 1=rojo)]
X = np.array([[5, 0], [6, 0], [7, 0], [10, 1], [12, 1], [14, 1]])

# Y: Etiquetas [Tipo de fruta (0=manzana, 1=naranja)]
Y = np.array([0, 0, 0, 1, 1, 1])

# 2. Creación y Entrenamiento del Modelo
# Crear el clasificador
model = DecisionTreeClassifier()

# Entrenar el modelo
model.fit(X, Y)

# 3. Predicción
# Predecir el tipo de fruta para una muestra [8 cm, 1 (rojo)]
sample_to_predict = np.array([[8, 1]])
prediction = model.predict(sample_to_predict)

fruit_type = 'Manzana' if prediction[0] == 0 else 'Naranja'
print(f"El tipo de fruta predicho para [Tamaño=8 cm, Color=Rojo] es: {fruit_type}")

# 4. Visualización del Árbol
plt.figure(figsize=(10, 8))
tree.plot_tree(
    model, 
    filled=True, 
    feature_names=["Tamaño", "Color"], 
    class_names=["Manzana", "Naranja"],
    rounded=True
)
plt.title("Árbol de Decisión para Clasificación de Frutas")
plt.show()