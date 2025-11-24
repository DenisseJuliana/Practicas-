import cv2

# Cargar la imagen
imagen = cv2.imread('pasillo.jpg')

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar bordes con el algoritmo Canny
bordes = cv2.Canny(gris, 100, 200)

# Mostrar la imagen original y los bordes detectados
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Bordes Detectados', bordes)

cv2.waitKey(0)
cv2.destroyAllWindows()