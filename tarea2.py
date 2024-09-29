import numpy as np
import skimage
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    """
    Esta función recibe una imagen y devuelve otra imagen recortada.

    Args:
      ruta_img (str): Ruta de la imagen original que se desea recortar.
      ruta_img_crop (str): Ruta donde se guardará la imagen recortada.
      x_inicial (int): Coordenada x inicial del área de recorte.
      x_final (int): Coordenada x final del área de recorte.
      y_inicial (int): Coordenada y inicial del área de recorte.
      y_final (int): Coordenada y final del área de recorte.

    Return
      None
    """
    try:
        # Abrimos la imagen
        image = cv2.imread(ruta_img)

        # Obtenemos la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardamos la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


# Función para cargar imagen. Devuelve la imagen o nulo si es válida o no.
def cargar_imagen(ruta):
    # Validamos que la imagen exista
    try:
        # Cargamos la imagen
        imagen1 = cv2.imread(ruta)
        return imagen1
    except Exception as e:
        print(f"Algó falló al intentar abrir la imagen {ruta}", str(e))
        return None
    
# Función para obtener el tamaño de la imagen
def obtener_tamaño(imagen):
    # Validamos que la imagen no sea nula
    if imagen is not None:
        return imagen.shape

# Función para obtener el minimo tamaño necesario para que ambas imagenes tengan el mismo tamaño y sean cuadradas
def obtener_tamaño_cuadrado(tamaño_1, tamaño_2):
    try:
        # Valida que las imagenes no sean nulas y que tengan tamaño valido  
        if tamaño_1 is not None and tamaño_2 is not None:
            
            # Obtenemos el tamaño minimo de la fila de las dos imagenes
            min_filas = min(tamaño_1[0],tamaño_2[0])
            
            # Obtenemos el tamaño minimo de la fila de las dos imagenes
            min_columnas = min(tamaño_1[1],tamaño_2[1])
            
            # Obtenemos el tamaño minimo entre las filas y columnas
            return min(min_filas, min_columnas)
        return None
    except Exception as e:
        print("Algó falló al intentar obtener el tamaño cuadrado", str(e))
        return None

# Función para transponer las filas y columnas de la matriz
def trasponer_matriz(img):
    try:
        # Transponemos únicamente la fila y columna.
        return np.transpose(img, (1,0,2))
    except Exception as e:
        print("Algó falló al intentar trasponer la imagen", str(e))

# Función para crear una imagen a partir de la matriz
def crear_imagen_matriz(nombre, matriz):
    try:
        # Cargamos la imagen
        cv2.imwrite(nombre, matriz)
    except Exception as e:
        print(f"Algó falló al intentar crear la imagen", str(e))
 
# Función para convertir a escala de grises.  
def convertir_a_grises(nombre, matriz):
    try:
        # Calculamos el promedio simple de los valores RGB para cada píxel
        matriz_gris = np.mean(matriz, axis=2)

        # Nos aseguramos de que los valores estén en el rango [0, 255] 
        matriz_gris_normalizada = matriz_gris.astype(np.uint8)

        # Creamos una nueva imagen a partir de la matriz en escala de grises
        imagen_gris = Image.fromarray(matriz_gris_normalizada)

        # Guardamos la nueva imagen
        imagen_gris.save(nombre)

        return matriz_gris_normalizada
    except Exception as e:
        print("Algo falló al convertir la imagen", str(e))

# Función para calcular el detemrinante    
def calcular_determinante(matriz): 
    try:      
        # Retornamos el determinante de la matriz  
        return np.linalg.det(matriz)

    except Exception as e:
        print("Algo falló al calcular el determinante:", str(e))
        return None

# Multiplicamos por un escalar una matriz que corresponde a una img y guarda lo nuevo
def aplicar_contraste(nombre, valor, matriz):
    try:  
        matriz_por_a = valor * matriz

        matriz_normalizada =  np.clip(matriz_por_a, 0, 255).astype(np.uint8)

        # Creamos una nueva imagen a partir de la matriz en escala de grises
        imagen_gris = Image.fromarray(matriz_normalizada, mode="L")

        # Guardamos la nueva imagen
        imagen_gris.save(nombre)

        return matriz_normalizada   
    except Exception as e:
        print("Algo falló al aplicar el contraste:", str(e))
        return None
    


def obtener_identidad(matriz):
    try:
        if matriz.ndim == 2 and matriz.shape[0] == matriz.shape[1]:
            return np.eye(matriz.shape[0])
            
        return "Debe ser una matriz cuadrada"
    except Exception as e:
        print("Algo falló al intentar encontrar la identidad:", str(e))
        return None

# Enviamos dos matrices que se multiplicaran, se crea una img y se guarda con
# un cierto nombre
def rotar_img(matriz1, matriz2, nombre):
    try:  

        # Para hacer la multiplicación, la primera debe tener tantas filas
        # como columnas tenga la segunda para poder realizar esta operación.

        if matriz1.shape[0] != matriz2.shape[1]:
            return "La primera debe tener tantas filas como columnas tiene la segunda"

        # Multiplicamos las dos matrices
        matriz_res = np.dot(matriz1, matriz2)

        # Creamos una nueva imagen a partir de la matriz en escala de grises
        imagen_volteada = Image.fromarray(matriz_res.astype(np.uint8), mode="L")
        imagen_volteada.save(nombre)
        return True   
    except Exception as e:
        print("Algo falló al aplicar el contraste:", str(e))
        return False 

# Recibe una matriz, crea el negativo de la matriz y guarda la imagen con el nombre pasado
def negativo_imagen(matriz, nombre):
    try:  

        # Restamos
        matriz_res = 255 - matriz

        # Creamos una nueva imagen a partir de la matriz en escala de grises
        imagen_volteada_1 = Image.fromarray(matriz_res.astype(np.uint8), mode="L")
        imagen_volteada_1.save(nombre)

        return True   
    except Exception as e:
        print("Algo falló al aplicar el contraste:", str(e))
        return False 

def run():

    try:

        # Cargamos las dos imagenes e imprimimos sus tamaños
        ruta_1 = 'img_python.jpg'
        ruta_2 = 'img_rattlesnake.jpg'   
        img_1 = cargar_imagen(ruta_1)
        tamaño_1 = obtener_tamaño(img_1)
        print(f"Tamaño {ruta_1} es de {tamaño_1}")
        img_2 = cargar_imagen(ruta_2)
        tamaño_2 = obtener_tamaño(img_2)
        print(f"Tamaño {ruta_2} es de {tamaño_2}")
        
        # Obtenemos el tamaño minimo de la fila de las dos imagenes
        tamaño_n = obtener_tamaño_cuadrado(tamaño_1, tamaño_2)
        print(f"Imagen cuadrada de n = {tamaño_n}")

        # Recortamos la primera imagen
        recortar_imagen_v2(ruta_1, 'img_1_cuadrada.jpg', 0, tamaño_n, 0, tamaño_n)

        # Recortamos la segunda imagen
        recortar_imagen_v2(ruta_2, 'img_2_cuadrada.jpg', 0, tamaño_n, 0, tamaño_n)


        # Para realizar ciertas operaciones entre las dos imagenes, como multiplicación de matrices, 
        # es necesario que estas tengan la misma dimensión. Por esta razón las hacemos cuadradas.

        # Seleccionamos una de las imagenes recortadas y la cargo
        img_2_cuadrada = cargar_imagen('img_2_cuadrada.jpg')
        img_1_cuadrada = cargar_imagen('img_1_cuadrada.jpg')

        # Imprimimos su tamaño
        print(f"Tamaño de una de las matrices recortadas: {obtener_tamaño(img_2_cuadrada)}")

        # La mostramos como matriz
        print(f"Mostrarla como matriz:\n")
        print(img_2_cuadrada)

        # Trasponemos img_1_cuadrada
        print("Transponer img_1_cuadrada y mostrarla como matriz:\n")
        img_1_cuadrada_traspuesta=trasponer_matriz(img_1_cuadrada)
        
        # La imprimimos
        print(img_1_cuadrada_traspuesta)

        # Generamos una nueva imagen a partir de la traspuesta
        crear_imagen_matriz('img_1_cuadrada_trapuesta.jpg', img_1_cuadrada_traspuesta)

        # Trasponemos img_2_cuadrada
        print("Transponer img_2_cuadrada y mostrarla como matriz:\n")
        img_2_cuadrada_traspuesta=trasponer_matriz(img_2_cuadrada)

        # La imprimimos
        print(img_2_cuadrada_traspuesta)

        # Generamos una nueva imagen a partir de la traspuesta
        crear_imagen_matriz('img_2_cuadrada_trapuesta.jpg', img_2_cuadrada_traspuesta)
        
        # Convertimos la img_1_cuadrada a escala de grises
        img_1_cuadrada_gris = convertir_a_grises('img_1_cuadrada_gris.jpg',img_1_cuadrada)

        # Convertimos la img_2_cuadrada a escala de grises
        img_2_cuadrada_gris = convertir_a_grises('img_2_cuadrada_gris.jpg',img_2_cuadrada)

        # Pasamos a calcular los determinantes
        det_img_1_gris=calcular_determinante(img_1_cuadrada_gris)
        det_img_2_gris=calcular_determinante(img_2_cuadrada_gris)
        print(f'Determinante de img_1_cuadrada_gris.jpg es de {det_img_1_gris}')
        print(f'Determinante de img_2_cuadrada_gris.jpg es de {det_img_2_gris}')

        # Para que una matriz se invertible, su determinante debe ser distinto de 0
        img_1_invertible = " si " if det_img_1_gris != 0 else " no"
        img_2_invertible = " si " if det_img_2_gris != 0 else " no"

        # Imprimimos si cada matriz es invertible o no
        print(f"La matriz de la imagen img_1_cuadrada_gris.jpg {img_1_invertible} es invertible")
        print(f"La matriz de la imagen img_2_cuadrada_gris.jpg {img_2_invertible} es invertible")

        # Ahora pasamos a calcular la inversa, si es que tienen
        if det_img_1_gris != 0:
            inversa_img_1 = np.linalg.inv(img_1_cuadrada_gris)
            print(f"La inversa de img_1_cuadrada_gris.jgp es {inversa_img_1}")
        
        if det_img_2_gris != 0:
            inversa_img_2 = np.linalg.inv(img_2_cuadrada_gris)
            print(f"La inversa de img_2_cuadrada_gris.jgp es {inversa_img_2}")

        # Pasamos a multiplicar la matriz de grieses de la imagen 1 por un escalar a1 = 5
        valor = 5
        matriz_por_a1 = aplicar_contraste(f"img_1_constraste_{valor}.jpg", valor, img_1_cuadrada_gris )
        print(f"Matriz resultante de la multiplicación por {valor} de la matriz de la img 1 {matriz_por_a1}")
    
        # Pasamos a multiplicar la matriz de grieses de la imagen 1 por un escalar a2 = 0.5
        valor = 0.1
        matriz_por_a2 = aplicar_contraste(f"img_1_contraste_{valor}.jpg", valor, img_1_cuadrada_gris )
        print(f"Matriz resultante de la multiplicación por {valor} de la matriz de la img 2 {matriz_por_a2}")
        
        """ 
        Cuanto más se aleja el valor a de 1 se distorsiona la imagen original
        perdiendose el contorno del objeto y su forma base

        En el caso que 0 < a < 1 se va oscureciendo cada vez más la imagen a medida que se acerca 
        el valor a o alpha a 0
        """
        
        # Ahora pasamos a comprobar que la multiplicación no es conmutativa
        identidad = obtener_identidad(img_1_cuadrada_gris)

        matriz_w = np.fliplr(identidad)

        # El resultado de esta rotación es una reflexión vertical
        rotar_img(matriz_w, img_1_cuadrada_gris, "w_por_matriz.jpg")


        # Ahora hacemos la otra multiplicación, y da una reflexión horizontal
        rotar_img(img_1_cuadrada_gris, matriz_w, "matriz_por_w.jpg")

        # Ahora obtenemos la matriz negativa, y generamos una imagen negativa
        negativo_imagen(img_1_cuadrada_gris, "negativo_img_1_gris.jpg")

        print("Fin del programa")
    except Exception  as e:
        print("Ocurrio un error en la ejecución del programa")

run()