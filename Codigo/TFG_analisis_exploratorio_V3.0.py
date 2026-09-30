# -*- coding: utf-8 -*-
"""
TFG: Machine Learning para el análisis de datos astronómicos
Grado en Matemáticas
Autor: David Díez Roshan
Tutora: Leire Beitia Antero

Descripción:
Este script realiza un flujo completo de análisis de datos astronómicos (SDSS):
1. Carga y Análisis Exploratorio de Datos (EDA).
2. Limpieza y Preprocesamiento.
3. Aprendizaje Supervisado (Random Forest y SVM).
4. Aprendizaje No Supervisado (K-Means y PCA).
"""

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías de Machine Learning (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuración de gráficos
sns.set_theme(style="whitegrid")

# --- 2. CARGA Y ANÁLISIS EXPLORATORIO ---
print("\n=== 1. CARGA DE DATOS ===")
# Cargamos forzando los IDs como texto para evitar errores de lectura,
# aunque tras el análisis preliminar decidimos no usarlos por baja calidad.
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

print(f"Datos cargados: {datos.shape[0]} observaciones, {datos.shape[1]} variables.")

print("\n=== 2. ANÁLISIS DE LA VARIABLE OBJETIVO ===")
conteo_clases = datos['class'].value_counts()
print(conteo_clases)

# Gráfico 1: Distribución de Clases
plt.figure(figsize=(8, 6))
sns.barplot(x=conteo_clases.index, y=conteo_clases.values, hue=conteo_clases.index, legend=False, palette="viridis")
plt.title("Distribución de Objetos Celestes (SDSS)", fontsize=14)
plt.ylabel("Número de Observaciones")
plt.xlabel("Clase")
plt.savefig("1_distribucion_clases.png")
print(">> Gráfico guardado: '1_distribucion_clases.png'")

# --- 3. LIMPIEZA Y PREPROCESAMIENTO ---
print("\n=== 3. LIMPIEZA Y SELECCIÓN DE VARIABLES ===")

# Selección de variables físicas (Petición Tutora)
# Eliminamos metadatos (IDs, fechas) y coordenadas (alpha, delta) para evitar sesgo espacial.
cols_fisicas = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
datos_limpios = datos[cols_fisicas].copy()

print("Variables seleccionadas para el análisis:")
print(datos_limpios.columns.tolist())

# Separación X (Características) e y (Etiqueta)
X = datos_limpios.drop(columns='class')
y = datos_limpios['class']

# División Train/Test (Estratificada por el desbalanceo de clases)
print("\nDividiendo datos en Entrenamiento (70%) y Prueba (30%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)
print(f"Entrenamiento: {X_train.shape[0]} muestras.")
print(f"Prueba:        {X_test.shape[0]} muestras.")

# --- 4. APRENDIZAJE SUPERVISADO: RANDOM FOREST ---
print("\n=== 4. MODELO 1: RANDOM FOREST (RF) ===")

# Random Forest no requiere escalado estricto, pero maneja bien datos crudos.
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
print("Entrenando Random Forest...")
rf_model.fit(X_train, y_train)

# Predicción
y_pred_rf = rf_model.predict(X_test)

# Evaluación
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"--> Precisión (Accuracy) RF: {acc_rf:.4f}")
print("\nReporte de Clasificación (RF):")
print(classification_report(y_test, y_pred_rf))

# --- 5. APRENDIZAJE SUPERVISADO: SVM ---
print("\n=== 5. MODELO 2: SUPPORT VECTOR MACHINE (SVM) ===")

# SVM REQUIERE escalado de datos (StandardScaler)
print("Escalando datos para SVM...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(random_state=42, kernel='rbf')
print("Entrenando SVM (Esto puede tardar unos minutos)...")
svm_model.fit(X_train_scaled, y_train)

# Predicción
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluación
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"--> Precisión (Accuracy) SVM: {acc_svm:.4f}")
print("\nReporte de Clasificación (SVM):")
print(classification_report(y_test, y_pred_svm))

# --- 6. APRENDIZAJE NO SUPERVISADO: K-MEANS ---
print("\n=== 6. MODELO 3: K-MEANS CLUSTERING ===")

# Usamos todo el dataset (X) escalado, sin etiquetas
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

# Buscamos 3 clusters (esperando encontrar Galaxy, Star, Qso)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
print("Ejecutando K-Means...")
kmeans.fit(X_scaled_full)
labels_kmeans = kmeans.labels_

# Tabla de Contingencia (Crosstab)
print("\nTabla de Contingencia (Realidad vs K-Means):")
tabla_contingencia = pd.crosstab(y, labels_kmeans, 
                                 rownames=['Realidad'], 
                                 colnames=['Cluster K-Means'])
print(tabla_contingencia)

# --- 7. ANÁLISIS VISUAL: PCA ---
print("\n=== 7. REDUCCIÓN DE DIMENSIONALIDAD (PCA) ===")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_full)

# DataFrame para facilitar el ploteo
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Clase Real'] = y.values
df_pca['Cluster KMeans'] = labels_kmeans

# Tomamos una muestra para no saturar el gráfico
df_sample = df_pca.sample(n=5000, random_state=42)

# Gráfico 2: PCA con Etiquetas Reales
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_sample, x='PC1', y='PC2', hue='Clase Real', palette='deep', alpha=0.6)
plt.title("PCA: Estructura Real de los Datos", fontsize=14)
plt.savefig("2_pca_real.png")
print(">> Gráfico guardado: '2_pca_real.png'")

# Gráfico 3: PCA con Clusters K-Means
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_sample, x='PC1', y='PC2', hue='Cluster KMeans', palette='viridis', alpha=0.6)
plt.title("PCA: Clusters asignados por K-Means", fontsize=14)
plt.savefig("3_pca_kmeans.png")
print(">> Gráfico guardado: '3_pca_kmeans.png'")

print("\n=== EJECUCIÓN COMPLETADA CON ÉXITO ===")