import cv2
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def cargarDatos(fase, numeroCategorias, limite, width, height):
    imagenesCargadas = []
    valorEsperado = []

    for categoria in range(0, numeroCategorias):
        for idImagen in range(0, limite[categoria]):
            ruta = fase + str(categoria) + "/" + str(categoria) + "_" + str(idImagen) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Asegurarse de que la profundidad de píxeles sea compatible (CV_8U)
            imagen = cv2.normalize(imagen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)  # Convertir a RGB si estás usando imágenes en color
            imagen = cv2.resize(imagen, (width, height))
            
            # Elimina la línea de aplanamiento
            # imagen = imagen.flatten()
            
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    
    imagenes_entrenamiento = np.array(imagenesCargadas)
    valores_esperados = np.array(valorEsperado)

    print("CANTIDAD DE IMAGENES", len(imagenes_entrenamiento))
    print("CANTIDAD DE VALORES", len(valores_esperados))

    return imagenes_entrenamiento, valores_esperados

width = 256
height = 256
pixeles = width * height
num_channels = 3  # Cambia a 3 para imágenes en color
img_shape = (width, height, num_channels)
num_clases = 4
cantidad_datos_entrenamiento = [4000, 4000, 4000, 4000]
cantidad_datos_pruebas = [2000, 2000, 2000, 2000]

##Carga de los datos
imagenes, probabilidades = cargarDatos("dataset/tratado/train/", num_clases, cantidad_datos_entrenamiento, width, height)

# Crear modelo VGG19 preentrenado
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(width, height, num_channels))

# Congelar capas convolucionales
for layer in base_model.layers:
    layer.trainable = False

# Agregar capas personalizadas al modelo
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_clases, activation="softmax"))

# Compilar el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Mostrar resumen del modelo
model.summary()

# Entrenar el modelo
model.fit(x=imagenes, y=probabilidades, epochs=5, batch_size=5)

# Pruebas
imagenes_prueba, probabilidades_prueba = cargarDatos("dataset/tratado/test/", num_clases, cantidad_datos_pruebas, width, height)
resultados = model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
print("METRIC NAMES", model.metrics_names)
print("RESULTADOS", resultados)

# Guardar el modelo
ruta = "models/model_vgg19.h5"
model.save(ruta)

# Estructura de la red
model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)

scnn_pred = model.predict(imagenes_prueba, batch_size=5, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)