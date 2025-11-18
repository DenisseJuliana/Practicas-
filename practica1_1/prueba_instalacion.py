import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame(data, columns=["Valor"])

plt.plot(df["Valor"])
plt.title("Gráfico de prueba")
plt.show()

print("Instalación exitosa!")
