#Imports

#Librería para Inteligencia Artificial
import tensorflow as tf
#Para arreglos numéricos
import numpy as np

#Inicializamos Inputs y Outputs
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Uso de Keras para hacer redes neuronales de manera simple.
#Inicializamos una capa de tipo densa. Una capa densa tiene conexiones 
#de una neurona hacia todas las demás de la siguiente capa
#Indicamos que solo tiene una salida en units y en input_shape que solo tenemos una entrada con una neurona
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

#Necesitamos ahora meter las neuronas en capas
modelo = tf.keras.Sequential([capa])

#Ahora con el modelo listo, el siguiente paso es compilarlo para prepararlo para entrenarlo
#Especificamos cómo queremos que procese ciertas matemáticas
#Indicaremos el optimizador y la función de pérdida
#Para el optimizador usaremos uno llamado Adam, que ajusta los pesos de manera eficiente para que aprenda y no desaprenda
#El valor numérico será la tasa de aprendizaje, es decir, cada cuanto va a ir ajustando
#Para el optimizador usamos error cuadrático medio -> Poca cantidad de errores grandes es mejor que muchos pequeños.
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Usamos la función fit() para entrenar. Indicamos inputs y outputs y cuantas veces queremos que de.

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])