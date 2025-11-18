import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Definición de Datos
# X: Tamaño de la casa (m²)
X = np.array([[50], [60], [80], [100], [120]])

# Y: Precio de la casa (miles de pesos)
Y = np.array([100, 120, 150, 180, 220])

# Creación y Entrenamiento del Modelo
model = LinearRegression()
model.fit(X, Y)

# Predicción
area_a_predecir = np.array([[90]])
predicted_price = model.predict(area_a_predecir)

print(f"El precio predicho para una casa de 90 m² es: {predicted_price[0]:.2f} mil pesos")

# Visualización
plt.figure(figsize=(8, 6))
# Dibuja los puntos de datos originales
plt.scatter(X, Y, color='blue', label='Datos Reales')
# Dibuja la recta de regresión aprendida por el modelo
plt.plot(X, model.predict(X), color='red', label='Línea de Regresión')
# Marca la predicción
plt.scatter(area_a_predecir, predicted_price, color='green', marker='o', s=100, label=f'Predicción ({area_a_predecir[0][0]} m²)')

plt.xlabel('Tamaño de la Casa ($m^2$)')
plt.ylabel('Precio de la Casa (mil pesos)')
plt.title('Regresión Lineal: Predicción del Precio de Casas')
plt.legend()
plt.grid(True)
plt.show()