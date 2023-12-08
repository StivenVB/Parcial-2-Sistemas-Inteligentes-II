import base64
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def cargarDatos(fase, numeroCategorias, limite, width, height):
    imagenesCargadas = []
    valorEsperado = []
    indiceImagen = 4000 if fase == "tratado/test/" else 0

    for categoria in range(0, numeroCategorias):
        for idImagen in range(indiceImagen, limite[categoria]):
            ruta = fase + str(categoria) + "/" + str(categoria) + "_" + str(idImagen) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width, height))
            imagen = imagen / 255.0  # Normalizar los valores de píxeles
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
num_channels = 1
img_shape = (width, height, num_channels)
num_clases = 4
cantidad_datos_entrenamiento = [4000, 4000, 4000, 4000]
cantidad_datos_pruebas = [6000, 6000, 6000, 6000]

imagenes, probabilidades = cargarDatos("tratado/train/", num_clases, cantidad_datos_entrenamiento, width, height)

model = models.Sequential()

# Capa de entrada
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

# Capa de salida
model.add(layers.Dense(num_clases, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60)

imagenes_prueba, probabilidades_prueba = cargarDatos("tratado/test/", num_clases, cantidad_datos_pruebas, width, height)
resultados = model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
print("METRIC NAMES", model.metrics_names)
print("RESULTADOS", resultados)

ruta = "modelos/modeloE.h5"
model.save(ruta)

model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)

cnn_pred = model.predict(imagenes_prueba, batch_size=60, verbose=1)
cnn_predicted = np.argmax(cnn_pred, axis=1)

cnn_cm = confusion_matrix(np.argmax(probabilidades_prueba, axis=1), cnn_predicted)

cnn_df_cm = pd.DataFrame(cnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)
sn.heatmap(cnn_df_cm, annot=True, annot_kws={"size": 12})
plt.show()

cnn_report = classification_report(np.argmax(probabilidades_prueba, axis=1), cnn_predicted)
print("CNN REPORT", cnn_report)