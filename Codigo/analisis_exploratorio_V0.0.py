# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 22:16:33 2025

@author: David
"""

import pandas as pd

# 1. Cargar los datos
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo)

# 2. Preprocesamiento: Eliminar columnas irrelevantes
# columnas de ID que no aportan información física
columnas_a_eliminar = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 
                       'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
                       
datos_limpios = datos.drop(columns = columnas_a_eliminar)

print("--- Datos después de la limpieza (head) ---")
print(datos_limpios.head())
print("\n") 

print("--- Información de datos limpios (info) ---")
print(datos_limpios.info())
print("\n") 

# 3. Separar en Características (X) y Objetivo (y)
# X = Todas las columnas MENOS 'class'
# y = Solo la columna 'class'

X = datos_limpios.drop(columns = 'class')
y = datos_limpios['class']

# 4. Comprobar la separación
print("--- Primeras filas de X (Características) ---")
print(X.head())
print("\n") 

print("--- Primeras filas de y (Objetivo) ---")
print(y.head())
print("\n")

# Contamos cuántos hay de cada clase
print("--- Distribución de clases en y ---")
print(y.value_counts())

# vamos a balancear los datos de galaxias quasares y estrellas
# Importamos la función para dividir los datos
from sklearn.model_selection import train_test_split

# 5. Dividir los datos en Entrenamiento (train) y Prueba (test)
#  test_size=0.3 significa que guardamos un 30% para el test
#  random_state=42 es solo un número fijo para que la división 
#  sea "aleatoria" pero siempre la MISMA cada vez 
#  que ejecutes el código (¡esencial para el TFG!)
#  stratify=y es la clave: asegura que el 30% del test tenga 
#  la misma proporción de GALAXY/STAR/QSO que el 70% de train.

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

# 6. Comprobar la división
print(f"--- Tamaño de los conjuntos ---")
print(f"X_train (entrenamiento): {X_train.shape}")
print(f"X_test (prueba):         {X_test.shape}")
print(f"y_train (entrenamiento): {y_train.shape}")
print(f"y_test (prueba):         {y_test.shape}")
print("\n")

print("--- Distribución en y_train (Entrenamiento) ---")
print(y_train.value_counts(normalize=True)) # normalize=True nos da porcentajes
print("\n")

print("--- Distribución en y_test (Prueba) ---")
print(y_test.value_counts(normalize=True)) # normalize=True nos da porcentajes

# (Tu código anterior para cargar, limpiar y dividir datos va aquí arriba)
# ...
# print(y_test.value_counts(normalize=True))
# ...

# --- 7. APRENDIZAJE SUPERVISADO: MODELO 1 (RANDOM FOREST) ---
print("\n--- 7. Entrenando Modelo 1: Random Forest ---")

# Importamos el modelo (Clasificador de Bosque Aleatorio)
from sklearn.ensemble import RandomForestClassifier

# Importamos las métricas de evaluación
from sklearn.metrics import accuracy_score, classification_report

# 1. Crear el modelo
#    random_state=42 para que el modelo sea reproducible
modelo_rf = RandomForestClassifier(random_state=42)

# 2. Entrenar el modelo
#    Esto es el "aprendizaje": el modelo busca patrones en X_train 
#    para predecir y_train
print("Entrenando... (Esto puede tardar unos segundos)")
modelo_rf.fit(X_train, y_train)
print("¡Entrenamiento completado!")

# 3. Predecir con los datos de prueba
#    Ahora usamos el modelo entrenado para predecir las etiquetas 
#    del conjunto X_test (que nunca antes había visto)
y_pred_rf = modelo_rf.predict(X_test)

# 4. Evaluar el rendimiento
print("\n--- Resultados del Random Forest ---")

# Comparamos las predicciones (y_pred_rf) con las respuestas reales (y_test)
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy (Precisión Total): {accuracy:.4f}")

# Imprimimos el reporte de clasificación
# Aquí veremos el "precision", "recall" y "f1-score" para 
# cada clase (GALAXY, STAR, QSO), lo cual es clave 
# por nuestro desbalanceo.
print("\nReporte de Clasificación (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# (Tu código anterior del Random Forest va aquí arriba)
# ...
# print(classification_report(y_test, y_pred_rf))
# ...

# --- 8. PREPROCESAMIENTO EXTRA: ESCALADO DE DATOS (para SVM) ---
print("\n--- 8. Escalando datos para SVM ---")

# Importamos el escalador
from sklearn.preprocessing import StandardScaler

# 1. Creamos el objeto escalador
scaler = StandardScaler()

# 2. APRENDEMOS la media y la desviación de X_train
#    Usamos fit_transform() SÓLO en los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# 3. APLICAMOS esa misma transformación a X_test
#    (No volvemos a "aprender", solo "transformamos")
X_test_scaled = scaler.transform(X_test)

print("¡Datos escalados! X_train_scaled y X_test_scaled creados.")


# --- 9. APRENDIZAJE SUPERVISADO: MODELO 2 (SVM) ---
print("\n--- 9. Entrenando Modelo 2: Support Vector Machine (SVM) ---")

# Importamos el modelo (Clasificador de Vectores Soporte)
from sklearn.svm import SVC

# 1. Crear el modelo SVM
#    Usamos los datos ESCALADOS (X_train_scaled)
#    (Nota: El entrenamiento de SVM es MUCHO más lento que Random Forest)
print("Entrenando SVM... (Esto puede tardar varios minutos)")
modelo_svm = SVC(random_state=42)

# 2. Entrenar el modelo SVM
modelo_svm.fit(X_train_scaled, y_train)
print("¡Entrenamiento SVM completado!")

# 3. Predecir con los datos de prueba ESCALADOS
y_pred_svm = modelo_svm.predict(X_test_scaled)

# 4. Evaluar el rendimiento del SVM
print("\n--- Resultados del SVM ---")

# Comparamos las predicciones (y_pred_svm) con las respuestas reales (y_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy (Precisión Total) SVM: {accuracy_svm:.4f}")

# Imprimimos el reporte de clasificación
print("\nReporte de Clasificación (SVM):")
print(classification_report(y_test, y_pred_svm))

# ...
# print(classification_report(y_test, y_pred_svm))
# ...

# --- 10. APRENDIZAJE NO SUPERVISADO: MODELO 3 (K-MEANS) ---
print("\n--- 10. Entrenando Modelo 3: K-Means Clustering ---")

# Importamos K-Means
from sklearn.cluster import KMeans

# 1. Preparar los datos
#    Usaremos TODAS las 100,000 muestras de X (no X_train)
#    Y DEBEMOS escalarlas, ya que K-Means se basa en distancias.
scaler_kmeans = StandardScaler()
X_scaled_full = scaler_kmeans.fit_transform(X) # X es tu dataframe original de 100k filas

print("Datos escalados (100,000 muestras) listos para K-Means.")

# 2. Crear y Entrenar el modelo K-Means
#    n_clusters=3 --> Le pedimos que encuentre 3 grupos (sabemos que hay 3 clases)
#    n_init=10    --> Ejecuta el algoritmo 10 veces con diferentes inicios 
#                     y se queda con el mejor (esto es estándar)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

print("Entrenando K-Means... (Esto puede tardar un poco)")
kmeans.fit(X_scaled_full)
print("¡Entrenamiento K-Means completado!")

# 3. Obtener las etiquetas de los clusters
#    Esto nos da un array que dice a qué cluster (0, 1 ó 2) 
#    pertenece cada una de las 100,000 muestras
y_kmeans_pred = kmeans.labels_

# 4. Evaluar el Clustering (Comparar con la realidad)
#    ¡Esta es la parte clave!
#    Comparamos las etiquetas REALES (y) con las etiquetas 
#    ENCONTRADAS (y_kmeans_pred).
#    Usamos una "Tabla de Contingencia" (Crosstab).

print("\n--- Resultados de K-Means ---")
print("Comparación de etiquetas reales vs. clusters encontrados:")

# Necesitamos 'y' (nuestras etiquetas reales) y 'y_kmeans_pred'
# 'y' ya la teníamos de antes (las 100,000 etiquetas de GALAXY, STAR, QSO)

# Creamos la tabla de contingencia
# (Asegúrate de tener pandas importado como pd al inicio de tu script)
tabla_contingencia = pd.crosstab(y, y_kmeans_pred, 
                                 rownames=['Etiqueta Real'], 
                                 colnames=['Cluster Encontrado'])

print(tabla_contingencia)