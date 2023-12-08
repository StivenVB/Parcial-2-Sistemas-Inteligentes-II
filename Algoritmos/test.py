import os

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta = 'dataset/tratado/test/3'

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(ruta_carpeta)

# Filtrar solo los archivos de imagen (puedes ajustar esto según tus extensiones de archivo)
archivos_imagen = [archivo for archivo in archivos if archivo.endswith(('.png', '.jpg', '.jpeg'))]

# Función para renombrar las imágenes
def renombrar_imagenes(ruta_carpeta, archivos_imagen):
    for i, archivo in enumerate(archivos_imagen):
        # Construir el nuevo nombre de archivo
        nuevo_nombre = f'3_{i}.jpg'  # Puedes ajustar la extensión según tus archivos
        
        # Rutas de archivo antiguas y nuevas
        ruta_antigua = os.path.join(ruta_carpeta, archivo)
        ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)
        
        # Renombrar el archivo
        os.rename(ruta_antigua, ruta_nueva)
        #print(f'Renombrado: {archivo} -> {nuevo_nombre}')

# Llamar a la función para renombrar las imágenes
renombrar_imagenes(ruta_carpeta, archivos_imagen)
