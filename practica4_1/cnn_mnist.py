import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps # Necesario para procesar la imagen externa

# --- 1. Preparación de Datos ---
# Cargar y preprocesar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reformatear para CNN: Añadir dimensión de canal (28x28 a 28x28x1)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalizar los valores de píxel de 0-255 a 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# --- 2. Creación y Entrenamiento del Modelo CNN ---
print("Creando y entrenando el modelo...")

model = models.Sequential([
    # Capa Convolucional 1: 32 filtros
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Capa Convolucional 2: 64 filtros
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Capa Convolucional 3: 64 filtros
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Aplanar los datos 3D a 1D para las capas Dense
    layers.Flatten(),

    # Capa Dense 1
    layers.Dense(64, activation='relu'),

    # Capa Dense de Salida: 10 neuronas (una para cada dígito 0-9)
    layers.Dense(10, activation='softmax')
])

# Compilar y entrenar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo con 5 épocas
model.fit(train_images, train_labels, epochs=5)

# Evaluar la precisión con el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nPrecisión en el conjunto de test: {test_acc:.4f}")

# --- 3. Función de Predicción con Imagen Externa (MODIFICADA) ---
def procesar_imagen(ruta_imagen):
    """
    Función para cargar, preprocesar una imagen externa, y hacer una predicción.
    Incluye binarización y centrado para mejorar la compatibilidad con MNIST.
    """
    try:
        # 1. Cargar y procesar imagen inicial
        imagen = Image.open(ruta_imagen).convert("L") # Cargar y convertir a escala de grises
        imagen = imagen.resize((28, 28)) 
        imagen_array = np.array(imagen)

        # Inversión de color (Asegura dígitos BLANCOS sobre fondo NEGRO)
        # Si el fondo es blanco (promedio > 127), invertimos.
        if np.mean(imagen_array) > 127:
             imagen_array = 255 - imagen_array

        # 2. Aplicar Binarización (Umbral) y Centrado (CRUCIAL)
        
        # Binarización: Convierte a blanco (255) o negro (0) puro.
        umbral = 50 # Ajusta este valor si el dígito es muy fino o muy grueso
        imagen_binaria = np.where(imagen_array > umbral, 255, 0).astype(np.uint8)
        
        coords = np.argwhere(imagen_binaria > 0)
        
        if coords.size > 0:
            # Encontrar los límites del dígito
            (y_min, x_min) = coords.min(axis=0)
            (y_max, x_max) = coords.max(axis=0)

            # Recortar el dígito
            recorte = imagen_binaria[y_min:y_max+1, x_min:x_max+1]
            (h, w) = recorte.shape
            
            # Crear un nuevo canvas de 28x28 para centrar el dígito
            nuevo_canvas = np.zeros((28, 28), dtype=np.uint8)
            
            # Escalar el recorte para que el lado más largo no sea más de 20 píxeles
            escala = 20 / max(h, w, 1) # Evitar división por cero
            recorte_redimensionado = Image.fromarray(recorte).resize(
                (int(w * escala), int(h * escala)), Image.Resampling.LANCZOS
            )
            recorte_array = np.array(recorte_redimensionado)
            (h_new, w_new) = recorte_array.shape

            # Calcular las coordenadas para centrar
            x_offset = (28 - w_new) // 2
            y_offset = (28 - h_new) // 2
            
            # Pegar el dígito centrado en el nuevo canvas
            nuevo_canvas[y_offset:y_offset + h_new, x_offset:x_offset + w_new] = recorte_array
            imagen_final_array = nuevo_canvas
        else:
            imagen_final_array = imagen_binaria


        # 3. Mostrar la imagen procesada
        imagen_final_pil = Image.fromarray(imagen_final_array)
        plt.imshow(imagen_final_pil, cmap='gray')
        plt.title(f"Dígito procesado para '{ruta_imagen}'")
        plt.show()

        # 4. Convertir a array y normalizar
        imagen_final_array = imagen_final_array / 255.0 # Normalizar a 0-1
        
        # Reformatear a 1x28x28x1 para el modelo
        imagen_array_tensor = imagen_final_array.reshape(1, 28, 28, 1)

        # 5. Hacer predicción
        prediccion = model.predict(imagen_array_tensor, verbose=0)
        digito = np.argmax(prediccion)
        confianza = prediccion[0][digito] * 100

        # 6. Imprimir resultados
        print(f"\nPredicción de '{ruta_imagen}':")
        print(f"El dígito es un **{digito}** con **{confianza:.2f}%** de confianza.")
        
        print("\nProbabilidades para cada dígito (0-9):")
        for i, prob in enumerate(prediccion[0]):
            print(f"Dígito {i}: {prob:.2%}")

    except FileNotFoundError:
        print(f"\nError: El archivo '{ruta_imagen}' no fue encontrado. Asegúrate de que está en la misma carpeta o la ruta es correcta.")
    except Exception as e:
        print(f"\nError al procesar la imagen: {e}")



# --- 4. Ejecutar la Predicción ---
# ¡Asegúrate de que 'ejemplo2.jpg' con el número 8 esté en la misma carpeta!
ruta = "ejemplo2.jpg"
procesar_imagen(ruta)