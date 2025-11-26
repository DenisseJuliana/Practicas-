import pygame
import random
import pandas as pd
import os

# --- Constantes y Configuración Global ---
# Constante para el archivo de datos
CSV_FILE = 'game_data.csv'

pygame.init()

# Configuración de la pantalla
width, height = 600, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Juego de Colección de Datos")

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Jugador
player_size = 50
# Posición inicial del jugador (centrado en la parte inferior)
player_pos = [width // 2 - player_size // 2, height - player_size]
player_speed = 10

# Obstáculos
obstacle_size = 50
# Posición inicial del obstáculo (aleatoria en X, en la parte superior)
obstacle_pos = [random.randint(0, width - obstacle_size), 0]
obstacle_speed = 10

# Marcador y reloj
score = 0
clock = pygame.time.Clock()

# --- Funciones de Dibujo ---
def dibujar_jugador(pos):
    """Dibuja el jugador en la pantalla."""
    pygame.draw.rect(screen, BLACK, (pos[0], pos[1], player_size, player_size))

def dibujar_obstaculo(pos):
    """Dibuja el obstáculo en la pantalla."""
    pygame.draw.rect(screen, RED, (pos[0], pos[1], obstacle_size, obstacle_size))

# --- Funciones de Eventos y Control ---
def manejar_eventos():
    """Verifica los eventos del juego y devuelve True si el juego debe terminar."""
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            return True
    return False

def manejar_control_humano():
    """Gestiona el movimiento del jugador humano y devuelve la tecla presionada (0, K_LEFT, K_RIGHT)."""
    tecla_presionada = 0
    teclas = pygame.key.get_pressed()
    
    if teclas[pygame.K_LEFT] and player_pos[0] > 0:
        player_pos[0] -= player_speed
        tecla_presionada = pygame.K_LEFT
    elif teclas[pygame.K_RIGHT] and player_pos[0] < width - player_size:
        player_pos[0] += player_speed
        tecla_presionada = pygame.K_RIGHT
    
    # Si no se presiona nada, tecla_presionada sigue siendo 0 (Quieto)
    return tecla_presionada

def manejar_control_ia(modelo_ia):
    """Gestiona el movimiento controlado por la IA (Requiere un modelo IA definido)."""
    # Esta función asume que 'modelo_ia' tiene un método 'predecir_accion' que 
    # devuelve 0 (Izquierda), 1 (Quieto), o 2 (Derecha).
    
    # Esta lógica depende de que el objeto modelo_ia exista y esté bien definido.
    # En este script, solo la usaremos si se pasa un modelo_ia.
    try:
        accion = modelo_ia.predecir_accion(player_pos, obstacle_pos, obstacle_speed)
    except AttributeError:
        # En el contexto de solo acomodar el código, esta rama evita errores.
        return

    if accion == 0 and player_pos[0] > 0:
        player_pos[0] -= player_speed
    elif accion == 2 and player_pos[0] < width - player_size:
        player_pos[0] += player_speed

# --- Funciones de Lógica del Juego ---
def actualizar_obstaculo():
    """Actualiza la posición del obstáculo y lo reinicia si sale de pantalla."""
    global obstacle_pos, score
    
    if obstacle_pos[1] >= height:
        # Reiniciar obstáculo y aumentar puntuación
        obstacle_pos = [random.randint(0, width - obstacle_size), 0]
        score += 1
    else:
        # Mover obstáculo
        obstacle_pos[1] += obstacle_speed

def verificar_colision():
    """Comprueba si hay colisión entre el jugador y el obstáculo."""
    # Lógica de colisión de rectángulos
    return (player_pos[0] < obstacle_pos[0] + obstacle_size and
            player_pos[0] + player_size > obstacle_pos[0] and
            player_pos[1] < obstacle_pos[1] + obstacle_size and
            player_pos[1] + player_size > obstacle_pos[1])

def renderizar_juego():
    """Dibuja todos los elementos gráficos del juego."""
    screen.fill(WHITE)
    dibujar_jugador(player_pos)
    dibujar_obstaculo(obstacle_pos)
    
    # Mostrar puntuación
    fuente = pygame.font.SysFont(None, 35)
    texto = fuente.render(f"Puntuación: {score}", True, BLACK)
    screen.blit(texto, (10, 10))
    
    pygame.display.update()

# --- Funciones de Captura de Datos (Pandas) ---
def inicializar_csv():
    """Crea el archivo CSV con encabezados si no existe."""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=[
            'JugadorX', 'JugadorY', 'ObstaculoX', 'ObstaculoY',
            'Velocidad', 'DistanciaX', 'Izquierda', 'Quieto', 'Derecha'
        ])
        df.to_csv(CSV_FILE, index=False)

def guardar_datos(tecla_presionada):
    """Guarda los datos del estado actual del juego y la acción humana en el CSV."""
    datos = {
        # Estado del juego
        'JugadorX': player_pos[0],
        'JugadorY': player_pos[1],
        'ObstaculoX': obstacle_pos[0],
        'ObstaculoY': obstacle_pos[1],
        'Velocidad': obstacle_speed,
        'DistanciaX': abs(player_pos[0] - obstacle_pos[0]),
        
        # Acciones One-Hot Encoding
        'Izquierda': 1 if tecla_presionada == pygame.K_LEFT else 0,
        'Quieto': 1 if tecla_presionada == 0 else 0, # Tecla_presionada = 0 si no hubo movimiento
        'Derecha': 1 if tecla_presionada == pygame.K_RIGHT else 0
    }
    # Añadir los datos al CSV sin escribir el encabezado nuevamente
    pd.DataFrame([datos]).to_csv(CSV_FILE, mode='a', header=False, index=False)

# --- Bucle Principal del Juego ---
def game_loop(control_ia=False, modelo_ia=None):
    """Bucle principal del juego. control_ia=True activa el control por IA."""
    global player_pos, obstacle_pos, score
    
    # Resetear variables globales para un nuevo juego (si se llama game_loop múltiples veces)
    player_pos = [width // 2 - player_size // 2, height - player_size]
    obstacle_pos = [random.randint(0, width - obstacle_size), 0]
    score = 0
    
    juego_terminado = False
    inicializar_csv() # Corregido: Llamar a la función correctamente

    while not juego_terminado:
        # 1. Manejo de Eventos (Salir)
        juego_terminado = manejar_eventos()

        # 2. Control del Jugador (Humano o IA)
        if not control_ia:
            # Control humano: Captura el input y lo guarda
            tecla = manejar_control_humano()
            guardar_datos(tecla) 
        else:
            # Control IA: Requiere un modelo entrenado
            manejar_control_ia(modelo_ia)

        # 3. Actualización de la Lógica del Juego
        actualizar_obstaculo()
        
        # 4. Colisión y Fin del Juego
        juego_terminado = juego_terminado or verificar_colision()

        # 5. Renderizado
        renderizar_juego()
        
        # 6. Control de FPS
        clock.tick(30)
        
    pygame.quit()
    print(f"\nJuego terminado. Puntuación final: {score}")
    print(f"Los datos de la partida se guardaron en {CSV_FILE}")

if __name__ == "__main__":
    game_loop(control_ia=False) # Inicia en modo control humano para capturar datos