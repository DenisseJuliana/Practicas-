import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Carga de datos con rutas relativas
df_ratings = pd.read_csv("ratings.csv")
df_movies = pd.read_csv("movies.csv")

# Inspección inicial
print("Primeras filas de ratings:")
print(df_ratings.head())
print("\nPrimeras filas de movies:")
print(df_movies.head())

# Subconjunto de datos para las 10 primeras películas
df_ratings_subset = df_ratings[df_ratings['movieId'].isin(df_movies['movieId'].head(10))]

# Definición de características (X) y etiqueta (y)
X = df_ratings_subset[['userId', 'movieId']]
y = df_ratings_subset['rating']

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nError cuadrático medio (MSE) del modelo: {mse:.4f}")

# Predicción de calificación para un usuario y película específicos
user_id = 1
movie_id = 31
predicted_rating = model.predict([[user_id, movie_id]])[0]
print(f"\nLa calificación predicha para el usuario {user_id} sobre la película {movie_id} es: {predicted_rating:.2f}")