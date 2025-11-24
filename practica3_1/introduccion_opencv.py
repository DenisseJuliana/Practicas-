import cv2

# Cargar la imagen
imagen = cv2.imread('paisaje.jpg')

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen en escala de grises
cv2.imshow('Imagen en Escala de Grises', gris)

cv2.waitKey(0)
cv2.destroyAllWindows()