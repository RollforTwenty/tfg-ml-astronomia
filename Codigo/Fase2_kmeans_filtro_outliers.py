# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 02_kmeans_filtro_outliers.py
FASE DEL TFG: Fase 2 (Investigación de limitaciones No Supervisadas)
=============================================================================

DESCRIPCIÓN:
Este script documenta un experimento exploratorio temprano. Ante el bajo
rendimiento inicial del algoritmo K-Means, se planteó la hipótesis de que
la inercia intra-cluster estaba siendo sesgada por valores atípicos severos.

PROCEDIMIENTO:
1. Se estandarizan las características observacionales crudas.
2. Se aplica el algoritmo de detección de anomalías 'Local Outlier Factor' (LOF) 
   para identificar y purgar el 1% de las observaciones más alejadas de su 
   vecindad local.
3. Se re-entrena el algoritmo K-Means sobre el conjunto de datos purgado.

CONCLUSIÓN EXPERIMENTAL:
Los resultados (tabla de contingencia) demostraron que la eliminación del ruido 
no mejoraba la capacidad de K-Means para aislar las clases, corroborando que 
el fallo se debía a la topología no convexa de los datos y a la degeneración 
por distancia (ausencia de redshift), hipótesis que se desarrolla formalmente 
en la Fase 3 de la memoria del TFG.
=============================================================================
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
# ... (aquí sigue el resto de tu código)

# 1. CARGAR Y PREPARAR DATOS (Lo que te faltaba)
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Usamos las variables físicas estándar
cols_fisicas = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
datos_limpios = datos[cols_fisicas].copy()

X = datos_limpios.drop(columns='class')
y = datos_limpios['class']

# 2. ESCALAR DATOS (Creamos la variable X_scaled_full)
print("--- Escalando datos... ---")
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

# 3. DETECTAR Y ELIMINAR OUTLIERS
print("\n--- DETECTANDO OUTLIERS (Esto tarda unos segundos)... ---")
# LocalOutlierFactor busca puntos que están "lejos de la manada"
# contamination=0.01 significa que eliminaremos el 1% más raro
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
y_pred_outliers = lof.fit_predict(X_scaled_full)

# Creamos una máscara: True si es un dato normal, False si es outlier
mask_limpios = y_pred_outliers != -1 

X_sin_outliers = X_scaled_full[mask_limpios]
y_sin_outliers = y[mask_limpios]

print(f"Datos originales: {len(X_scaled_full)}")
print(f"Datos limpios:    {len(X_sin_outliers)} (Hemos borrado {len(X_scaled_full) - len(X_sin_outliers)} outliers)")

# 4. EJECUTAR K-MEANS SOBRE DATOS LIMPIOS
print("\n--- Ejecutando K-Means sobre datos limpios... ---")
kmeans_fixed = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_fixed = kmeans_fixed.fit_predict(X_sin_outliers)

# 5. RESULTADO FINAL
print("\nTabla de Contingencia (K-Means SIN outliers):")
print(pd.crosstab(y_sin_outliers, labels_fixed, 
                  rownames=['Realidad'], 
                  colnames=['Cluster K-Means']))