# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:25:16 2023

@author: DESARROLLO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,  LSTM#, Dropout
from keras.optimizers import Adam


# Cargar datos desde CSV
dataset = pd.read_csv('dataset_tsm.csv', sep=',')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalar los datos
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Reformatear los datos para LSTM (asumiendo que tienes múltiples series temporales)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Construir la red con LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1), activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Imprimir la arquitectura de la red
model.summary()

# Entrenar la red
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de entrenamiento
y_train_pred_prob = model.predict(X_train)
y_train_pred = (y_train_pred_prob > 0.5).astype(int)

# Matriz de confusión en el conjunto de entrenamiento
cm_train = confusion_matrix(y_train, y_train_pred)
print("Matriz de Confusión en el Conjunto de Entrenamiento:")
print(cm_train)

# Precisión del modelo en el conjunto de entrenamiento
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"Precisión del modelo en el Conjunto de Entrenamiento (Accuracy): {accuracy_train * 100:.2f}%")

precision_train = precision_score(y_train, y_train_pred)
print(f"Precisión del modelo en el Conjunto de Entrenamiento (Precision): {precision_train * 100:.2f}%")


# Evaluar el modelo en el conjunto de prueba
y_test_pred_prob = model.predict(X_test)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Matriz de confusión en el conjunto de prueba
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de Confusión en el Conjunto de Prueba:")
print(cm_test)

# Precisión del modelo en el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Precisión del modelo en el Conjunto de Prueba (Accuracy): {accuracy_test * 100:.2f}%")

precision_test = precision_score(y_test, y_test_pred)
print(f"Precisión del modelo en el Conjunto de Prueba (Precision): {precision_test * 100:.2f}%")

# Precisión del modelo en el conjunto de entrenamiento
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)

# Precisión del modelo en el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)

# Etiquetas y valores para el gráfico
labels = ['Accuracy (Entrenamiento)', 'Precision (Entrenamiento)', 'Accuracy (Prueba)', 'Precision (Prueba)']
values = [accuracy_train * 100, precision_train * 100, accuracy_test * 100, precision_test * 100]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.title('Comparación de Precisión y Accuracy del Modelo en Conjuntos de Entrenamiento y Prueba')
plt.ylabel('Precisión (%)')

# Mostrar los valores en las barras
for i, value in enumerate(values):
    plt.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom')

plt.show()

# Visualizar la matriz de confusión en el conjunto de entrenamiento
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
plt.title('Matriz de Confusión (Entrenamiento)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Visualizar la matriz de confusión en el conjunto de prueba
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.title('Matriz de Confusión (Prueba)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

#Incorporad 02.02.2024
# Precisión, Precision y F1-score del modelo en el conjunto de entrenamiento
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
print(f"Precisión del modelo en el Conjunto de Entrenamiento (Accuracy): {accuracy_train * 100:.2f}%")
print(f"Precisión del modelo en el Conjunto de Entrenamiento (Precision): {precision_train * 100:.2f}%")
print(f"F1-score del modelo en el Conjunto de Entrenamiento: {f1_train:.2f}")

# Evaluar el modelo en el conjunto de prueba
y_test_pred_prob = model.predict(X_test)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Matriz de confusión en el conjunto de prueba
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de Confusión en el Conjunto de Prueba:")
print(cm_test)

# Precisión, Precision y F1-score del modelo en el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
print(f"Precisión del modelo en el Conjunto de Prueba (Accuracy): {accuracy_test * 100:.2f}%")
print(f"Precisión del modelo en el Conjunto de Prueba (Precision): {precision_test * 100:.2f}%")
print(f"F1-score del modelo en el Conjunto de Prueba: {f1_test:.2f}")
#Fin 02.02.2024

# Datos de entrada para la prueba
#new_data = np.array([[18.48,20.95,24.53,21.52]])
new_data = np.array([[18.48,20.95,24.53,21.52]])
# Escalar los nuevos datos
new_data_scaled = sc_X.transform(new_data)

# Reformatear los datos para LSTM
new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

# Realizar la predicción
prediction = model.predict(new_data_reshaped)

# Convertir la predicción a 0 o 1 basado en el umbral de 0.5
prediction_class = (prediction > 0.5).astype(int)

# Imprimir la predicción
print(f"Predicción del modelo para las nuevas entradas: {prediction_class[0, 0]}")
print(prediction)