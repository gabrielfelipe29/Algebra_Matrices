import numpy as np
import skimage
import cv2
import os
import matplotlib.pyplot as plt

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
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


# Funcion para cargar imagen. Devuelve la imagen o nulo si es valida o no.
def cargar_imagen(ruta):
    # Valida que la imagen exista
    try:
        # Cargo la imagen
        imagen1 = cv2.imread(ruta)
        return imagen1
    except Exception as e:
        print(f"Algó falló al intentar abrir la imagen {ruta}", str(e))
        return None
    
# Funcion para obtener el tamaño de la imagen
def obtener_tamaño(imagen):
    # Valida que la imagen no sea nula
    if imagen is not None:
        return imagen.shape

# Funcion para obtener el minimo tamaño necesario para que ambas imagenes tengan el mismo tamaño y sean cuadradas
def obtener_tamaño_cuadrado(tamaño_1, tamaño_2):
    if tamaño_1 is not None and tamaño_2 is not None:
        # Obtiene el tamaño minimo de la fila de las dos imagenes
        min_filas=min(tamaño_1[0],tamaño_2[0])
        # Obtiene el tamaño minimo de la fila de las dos imagenes
        min_columnas=min(tamaño_1[1],tamaño_2[1])
        # Obtiene el tamaño minimo entre las filas y columnas
        return min(min_filas, min_columnas)

# Funcion para transponer las filas y columnas de la matriz
def trasponer_matriz(img):
    try:
        # Transpone únicamente la fila y columna.
        return np.transpose(img, (1,0,2))
    except Exception as e:
        print("Algó falló al intentar trasponer la imagen", str(e))

# Funcion para crear una imagen a partir de la matriz
def crear_imagen_matriz(nombre, matriz):
    try:
        # Cargo la imagen
        cv2.imwrite(nombre, matriz)
    except Exception as e:
        print(f"Algó falló al intentar crear la imagen", str(e))
 
# Funcion para convertir a escala de grises.  
def convertir_a_grises(nombre, matriz):
    try:
        imagen_gris = cv2.cvtColor(matriz, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(nombre, imagen_gris)

    except Exception as e:
        print("Algo falló al convertir la imagen", str(3))

# Funcion para calcular el detemrinante
def calcular_determinante(matriz): 
    # Verificar si la matriz es cuadrada
    print(f"{matriz.ndim} {matriz.shape[0] == matriz.shape[1]}")
    try:        
        return np.linalg.det([[np.asarray(matriz.shape[0])],[
                               np.asarray(matriz.shape[1])]])
        
    except Exception as e:
        print("Algo falló al calcular el determinante:", str(e))
        return None


def run():
    # Cargo las dos imagenes e imprimo sus tamaños
    ruta_1='img_python.jpg'
    ruta_2='img_rattlesnake.jpg'   
    img_1=cargar_imagen(ruta_1)
    tamaño_1=obtener_tamaño(img_1)
    print(f"Tamaño {ruta_1} es de {tamaño_1}")
    img_2=cargar_imagen(ruta_2)
    tamaño_2=obtener_tamaño(img_2)
    print(f"Tamaño {ruta_2} es de {tamaño_2}")
    
    # Obtiene el tamaño minimo de la fila de las dos imagenes
    tamaño_n = obtener_tamaño_cuadrado(tamaño_1, tamaño_2)
    print(f"Imagen cuadrada de n = {tamaño_n}")

    # Recorto la primera imagen
    recortar_imagen_v2(ruta_1, 'img_1_cuadrada.jpg',0,tamaño_n,0,tamaño_n)

    # Recorto la segunda imagen
    recortar_imagen_v2(ruta_2, 'img_2_cuadrada.jpg',0,tamaño_n,0,tamaño_n)

    # Para realizar ciertas operaciones entre las dos imagenes, como multiplicación de matrices, 
    # es necesario que estás tengan la misma dimensión. Por está razón las hacemos cuadradas.

    # Selecciono una de las imagenes recortadas y la cargo
    img_2_cuadrada=cargar_imagen('img_2_cuadrada.jpg')
    img_1_cuadrada=cargar_imagen('img_1_cuadrada.jpg')

    # Imprimo su tamaño
    print(f"Tamaño de una de las matrices recortadas: {obtener_tamaño(img_2_cuadrada)}")
    # La muestro como matriz
    print(f"Mostrarla como matriz:\n")
    print(img_2_cuadrada)


    # Traspongo img_1_cuadrada
    print("Transponer img_1_cuadrada y mostrarla como matriz:\n")
    img_1_cuadrada_traspuesta=trasponer_matriz(img_1_cuadrada)
    # La imprimo
    print(img_1_cuadrada_traspuesta)
    # Genero una nueva imagen a partir de la traspuesta
    crear_imagen_matriz('img_1_cuadrada_trapuesta.jpg', img_1_cuadrada_traspuesta)

    # Traspongo img_2_cuadrada
    print("Transponer img_2_cuadrada y mostrarla como matriz:\n")
    img_2_cuadrada_traspuesta=trasponer_matriz(img_2_cuadrada)
    # La imprimo
    print(img_2_cuadrada_traspuesta)
    # Genero una nueva imagen a partir de la traspuesta
    crear_imagen_matriz('img_2_cuadrada_trapuesta.jpg', img_2_cuadrada_traspuesta)
    
    # Convierte la img_1_cuadrada a escala de grises
    convertir_a_grises('img_1_cuadrada_gris.jpg',img_1_cuadrada)
    # Convierte la img_2_cuadrada a escala de grises
    convertir_a_grises('img_2_cuadrada_gris.jpg',img_2_cuadrada)

    # Cargo las imagenes de escala de grises
    img_1_cuadrada_gris=cargar_imagen('img_1_cuadrada_gris.jpg')
    img_2_cuadrada_gris=cargar_imagen('img_2_cuadrada_gris.jpg')
    print("Tamaño:",obtener_tamaño(img_1_cuadrada_gris))
    print("Tamaño:",obtener_tamaño(img_2_cuadrada_gris))

    det_img_1_gris=calcular_determinante(img_1_cuadrada_gris)
    det_img_2_gris=calcular_determinante(img_2_cuadrada_gris)
    print(f'Determinante de img_1_cuadrada_gris.jpg es de {det_img_1_gris}')
    print(f'Determinante de img_2_cuadrada_gris.jpg es de {det_img_2_gris}')

run()