from juego import game_loop, width, height # Importa la lógica del juego y las dimensiones
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import os
import time

# --- Constantes ---
MODELO_IA = "modelo_ia.h5"
# El archivo CSV debe coincidir con el nombre usado en juego.py
CSV_FILE = "game_data.csv" 

# --- Clase de Control de la IA ---
class ControlIA:
    """Clase que encapsula la lógica para predecir la acción del jugador."""
    def __init__(self, modelo, ancho, alto):
        self.modelo = modelo
        self.ancho = ancho
        self.alto = alto
        self.ultima_accion = "Ninguna"
        self.contador = 0 
        
    def predecir_accion(self, player_pos, obstacle_pos, velocidad):
        self.contador += 1
        distancia_x = player_pos[0] - obstacle_pos[0]
        
        # Lógica prioritaria (obstáculo cercano en Y) - Esta es una regla codificada
        # Si el obstáculo está en la mitad inferior de la pantalla...
        if obstacle_pos[1] > self.alto * 0.6:
            # Si el jugador está a la derecha del obstáculo, moverse a la izquierda
            if distancia_x > 0:
                self.ultima_accion = "Izquierda"
                return 0  # 0 = Izquierda
            # Si el jugador está a la izquierda del obstáculo, moverse a la derecha
            else:
                self.ultima_accion = "Derecha"
                return 2  # 2 = Derecha
        
        # --- Predicción con el modelo (si la regla codificada no aplica) ---
        
        # Datos de entrada para el modelo (6 características)
        datos_entrada = np.array([[
            player_pos[0], player_pos[1],
            obstacle_pos[0], obstacle_pos[1],
            velocidad, distancia_x
        ]])
        
        # Realizar la predicción
        prediccion = self.modelo.predict(datos_entrada, verbose=0)
        accion = np.argmax(prediccion)
        
        # Mapeo a acciones legibles
        if accion == 0:
            self.ultima_accion = "Izquierda"
        elif accion == 1:
            self.ultima_accion = "Quieto"
        else:
            self.ultima_accion = "Derecha"
            
        return accion

# --- Funciones de Keras y Entrenamiento ---

def crear_modelo():
    """Define la estructura de la Red Neuronal para la IA."""
    modelo = Sequential([
        # Capa de entrada con 6 neuronas (una por cada característica)
        Dense(32, input_shape=(6,), activation='relu'), 
        # Dropout para prevenir el sobreajuste (overfitting)
        Dropout(0.3), 
        # Capa oculta
        Dense(32, activation='relu'),
        # Capa de salida: 3 neuronas (Izquierda, Quieto, Derecha) con softmax para probabilidades
        Dense(3, activation='softmax')
    ])
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Usado para etiquetas One-Hot Encoded
        metrics=['accuracy']
    )
    return modelo

def entrenar_modelo():
    """Carga los datos del CSV, entrena el modelo y lo guarda."""
    if not os.path.exists(CSV_FILE):
        print(f"\n[ERROR] El archivo '{CSV_FILE}' no existe.")
        print("Ejecuta 'juego.py' varias veces en modo humano para crear el dataset.")
        exit()
        
    datos = pd.read_csv(CSV_FILE)
    if len(datos) < 50:
        print(f"\n[ADVERTENCIA] Necesitas más datos de entrenamiento (tienes {len(datos)}, se recomiendan al menos 50).")
        # No salimos, permitimos el entrenamiento con pocos datos
        
    print(f"\n[INFO] Cargando {len(datos)} registros para entrenamiento...")
    
    # Características (X): 6 columnas de estado del juego
    X = datos[['JugadorX', 'JugadorY', 'ObstaculoX', 'ObstaculoY', 'Velocidad', 'DistanciaX']].values
    # Etiquetas (y): 3 columnas de acción (Izquierda, Quieto, Derecha)
    y = datos[['Izquierda', 'Quieto', 'Derecha']].values
    
    modelo = crear_modelo()
    
    print("\n[INFO] Iniciando entrenamiento de la IA...")
    # Entrenar la red neuronal
    modelo.fit(X, y, epochs=200, batch_size=32, verbose=1)
    
    print(f"\n[INFO] Entrenamiento completado. Guardando modelo en '{MODELO_IA}'...")
    modelo.save(MODELO_IA)
    return modelo

def cargar_modelo():
    """Intenta cargar un modelo guardado; si no existe, entrena uno nuevo."""
    if os.path.exists(MODELO_IA):
        print(f"\n[INFO] Cargando modelo existente desde '{MODELO_IA}'...")
        # Agregar un manejo de errores si el archivo .h5 está corrupto
        try:
            return load_model(MODELO_IA)
        except Exception as e:
            print(f"[ERROR] Error al cargar el modelo: {e}. Entrenando uno nuevo...")
            return entrenar_modelo()
    
    # Si el archivo no existe, entrenar
    return entrenar_modelo()

# --- Función Principal ---

def main():
    if not os.path.exists(CSV_FILE):
        print("\n[ERROR] Por favor, ejecuta primero 'juego.py' en modo humano varias veces para generar datos.")
        return

    modelo = cargar_modelo()
    
    # Crear la instancia del controlador IA con el modelo entrenado y las dimensiones del juego
    control_ia = ControlIA(modelo, width, height)
    
    print("\n--- IA lista para JUGAR ---")
    print("Iniciando game_loop en modo IA. Presiona ESC o cierra la ventana para terminar.")
    
    # Iniciar el juego en modo IA
    game_loop(control_ia=True, modelo_ia=control_ia)

if __name__ == "__main__":
    main()