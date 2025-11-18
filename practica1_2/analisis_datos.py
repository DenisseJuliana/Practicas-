import pandas as pd
import matplotlib.pyplot as plt

# Carga los datos del archivo CSV
df = pd.read_csv("ventas.csv")

# Muestra las primeras filas del DataFrame para verificación
print(df.head())

# Crea un gráfico de línea
# 'marker="o"' añade círculos en cada punto de dato.
plt.plot(df["Producto"], df["Ventas"], marker="o")

# Personaliza el gráfico
plt.xlabel("Producto")
plt.ylabel("Cantidad Vendida")
plt.title("Tendencia de Ventas")

# Muestra el gráfico
plt.show()