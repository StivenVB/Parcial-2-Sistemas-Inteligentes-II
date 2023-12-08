import cv2
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def cargarDatos(fase,numeroCategorias,limite,width,height):
    imagenesCargadas=[]
    etiquetas=[]
    valorEsperado=[]
    indiceImagen = 4000 if fase == "tratado/test/" else 0
    for categoria in range(0,numeroCategorias):
        for idImagen in range(indiceImagen,limite[categoria]):
            ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen=cv2.imread(ruta)
            imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width,height))
            imagen=imagen.flatten()
            imagen=imagen/255
            imagenesCargadas.append(imagen)
            probabilidades=np.zeros(numeroCategorias)
            probabilidades[categoria]=1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento=np.array(imagenesCargadas)
    valoresEsperados=np.array(valorEsperado)
    return imagenesEntrenamiento,valoresEsperados

width=256
height=256
#Numero de neuronas de la cnn
img_size_flat=width*height
#Parametrizar la forma de imagenes
num_chanels=1
#RGB, HSV -> num_chanels=3
img_shape=(width,height,num_chanels)
num_clases=4
limiteImagenesEntrenamiento=[4000,4000,4000,4000]
imagenes,probabilidades=cargarDatos("tratado/train/",num_clases,limiteImagenesEntrenamiento,width,height)

model = Sequential()
# Capa entrada
model.add(InputLayer(input_shape=(img_size_flat,)))
# Reformar imagen
model.add(Reshape(img_shape))
# Capas convolucionales
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='capa_convolucion_1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='capa_convolucion_2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# Aplanar imagen
model.add(Flatten())
# Capa densa
model.add(Dense(128, activation='relu'))
# Capa salida
model.add(Dense(num_clases, activation='softmax'))

model.compile(optimizer="adam",
loss='categorical_crossentropy',
metrics=['accuracy']
)

#Entrenamiento del modelo
model.fit(x=imagenes,y=probabilidades,epochs=4,batch_size=100)

limiteImagenesPrueba=[6000,6000,6000,6000]
imagenesPrueba,probabilidadesPrueba=cargarDatos("tratado/test/",num_clases,limiteImagenesPrueba,width,height)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("{0}: {1:.2%}".format(model.metrics_names[1], resultados[1]))
#Carpeta y nombre del archivo como se almacenará el modelo
nombreArchivo='modelos/modeloA.h5'
model.save(nombreArchivo)
model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)

scnn_pred = model.predict(imagenesPrueba, batch_size=60, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)