import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np  # Para cálculos matemáticos
import pandas as pd  # Para manejar datos
import math  # Para funciones matemáticas
import datetime  # Para trabajar con fechas y horas
import platform  # Para información del sistema operativo
from sklearn.manifold import TSNE  # Para reducción de dimensiones
from sklearn.model_selection import train_test_split  # Para dividir datos

# Cargar datos de entrenamiento y prueba
train = pd.read_csv('train.csv')  # Cargar datos de entrenamiento
test = pd.read_csv('test.csv')  # Cargar datos de prueba
# Mostrar dimensiones de los conjuntos
"""
print('train:', train.shape)
print('test:', test.shape)
"""

X = train.iloc[:, 1:785]  # Obtener características
y = train.iloc[:, 0]  # Obtener etiquetas
X_test = test.iloc[:, 0:784]  # Obtener características del conjunto de prueba

"""
X_tsn = X / 255  # Normalizar datos
tsne = TSNE()  # Inicializar TSNE
tsne_res = tsne.fit_transform(X_tsn)  # Ajustar TSNE
plt.figure(figsize=(14, 12))  # Crear figura
plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y, s=2)  # Graficar
plt.xticks([])  # Ocultar marcas del eje x
plt.yticks([])  # Ocultar marcas del eje y
plt.colorbar()  # Mostrar barra de color
plt.show()  # Mostrar gráfica
"""

# Dividir datos en entrenamiento y validación
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1212)
"""print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_validation:', X_validation.shape)
print('y_validation:', y_validation.shape)
"""

# Redimensionar datos para el modelo
x_train_re = X_train.to_numpy().reshape(33600, 28, 28)  # Datos de entrenamiento
y_train_re = y_train.values  # Etiquetas de entrenamiento
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28)  # Datos de validación
y_validation_re = y_validation.values  # Etiquetas de validación
x_test_re = test.to_numpy().reshape(28000, 28, 28)  # Datos de prueba

# Guardar parámetros de imagen
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape  # Dimensiones de la imagen
IMAGE_CHANNELS = 1  # Canales de la imagen
print('IMAGE_WIDTH:', IMAGE_WIDTH)  # Imprimir ancho
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)  # Imprimir altura
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)  # Imprimir canales

pd.DataFrame(x_train_re[0])  # Mostrar primera imagen

plt.imshow(x_train_re[0], cmap=plt.cm.binary)  # Mostrar imagen
plt.show()  # Mostrar gráfica

# Mostrar ejemplos de entrenamiento
numbers_to_display = 25  # Número de ejemplos a mostrar
num_cells = math.ceil(math.sqrt(numbers_to_display))  # Celdas en la cuadrícula
plt.figure(figsize=(10, 10))  # Crear figura
for i in range(numbers_to_display):  # Iterar sobre ejemplos
    plt.subplot(num_cells, num_cells, i + 1)  # Crear subplot
    plt.xticks([])  # Ocultar marcas del eje x
    plt.yticks([])  # Ocultar marcas del eje y
    plt.grid(False)  # No mostrar cuadrícula
    plt.imshow(x_train_re[i], cmap=plt.cm.binary)  # Mostrar imagen
    plt.xlabel(y_train_re[i])  # Etiquetar imagen
plt.show()  # Mostrar gráfica
