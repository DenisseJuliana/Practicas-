import cv2

# Inicializa la cámara (0 es el índice de la cámara principal)
camara = cv2.VideoCapture(0)

# Carga el clasificador de rostros preentrenado (Haar Cascade)
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml') # Corregidas las comillas

while True:
    # Captura un cuadro de la cámara
    _, cuadro = camara.read() 
    
    # Convertir el cuadro a escala de grises
    gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY) 
    
    # Detección de rostros: Busca rostros en la imagen en escala de grises
    rostros = detector_rostros.detectMultiScale(
        gris, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # Dibuja un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in rostros:
        cv2.rectangle(cuadro, (x, y), (x + w, y + h), (0, 255, 0), 2) # Rectángulo verde
    
    # Muestra el video con la detección
    cv2.imshow('Detección de Rostros', cuadro) # Corregidas las comillas
    
    # Espera por la tecla 'q' para salir (waitKey(1) espera 1 milisegundo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
camara.release()
cv2.destroyAllWindows()