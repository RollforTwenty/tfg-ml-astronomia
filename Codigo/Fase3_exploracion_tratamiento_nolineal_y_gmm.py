# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 03_exploracion_nolineal_gmm.py
FASE DEL TFG: Fase 2 - Exploración de Algoritmos Avanzados
=============================================================================

DESCRIPCIÓN:
Este script documenta la experimentación algorítmica avanzada llevada a cabo 
para intentar mitigar los fallos de agrupación del algoritmo K-Means clásico. 
Se prueban dos hipótesis topológicas: la transformación no lineal del espacio 
de características y la relajación de la restricción de esfericidad geométrica.

PROCEDIMIENTO:
1. Ingeniería de Características No Lineales: Se generan nuevos índices 
   multiplicativos (u*g, g*r) para contrastarlos gráficamente contra los 
   índices de color sustractivos clásicos.
2. Modelado GMM (Gaussian Mixture Models): Se sustituye la partición de 
   Voronoi (K-Means) por un modelado de mezclas gaussianas. Al configurar 
   covariance_type='full', se permite al algoritmo ajustar elipses de 
   densidad no isotrópicas y rotadas, adaptándose teóricamente mejor al 
   'Stellar Locus'.

CONCLUSIÓN EXPERIMENTAL:
El análisis de las proyecciones PCA y las tablas de contingencia demostró 
que la flexibilización geométrica (GMM) es insuficiente para romper la 
fuerte degeneración fotométrica (solapamiento Estrella-Cuásar) si no se 
incorpora la dimensión de distancia (redshift).
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 1. CARGA
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. INGENIERÍA DE CARACTERÍSTICAS
# A. Lineales (Las clásicas)
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

# B. No Lineales (Sugerencia Tutora - Experimento)
# Probamos si multiplicar magnitudes separa mejor
datos['u*g'] = datos['u'] * datos['g']
datos['g*r'] = datos['g'] * datos['r']

# 3. VISUALIZACIÓN NO-LINEAL (¿Hay suerte?)
print("--- Generando gráfico No-Lineal... ---")
df_sample = datos.sample(2000, random_state=42)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df_sample, x='u-g', y='g-r', hue='class', s=10, alpha=0.6)
plt.title("Lineal (Clásico)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_sample, x='u*g', y='g*r', hue='class', s=10, alpha=0.6)
plt.title("No-Lineal (Experimental)")

plt.tight_layout()
plt.savefig("8_lineal_vs_nolineal.png")
print(">> Gráfico guardado: '8_lineal_vs_nolineal.png'")


# 4. EJECUCIÓN DE GMM (Gaussian Mixture)
# Usaremos las variables de COLOR estándar + Redshift, que son las que tienen sentido físico
features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos[features]
y = datos['class']

# Escalamos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Entrenando Gaussian Mixture (GMM)... ---")
# n_components=3 (buscamos 3 clases)
# covariance_type='full' (permite elipses de cualquier forma y rotación)
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
labels_gmm = gmm.fit_predict(X_scaled)

# 5. RESULTADOS
print("\n=== TABLA DE CONTINGENCIA (GMM) ===")
# Esta es la tabla que te dirá si hemos triunfado o no
print(pd.crosstab(y, labels_gmm, rownames=['Realidad'], colnames=['Cluster GMM']))

# Visualización del resultado GMM
# Usamos PCA para proyectar y ver qué ha hecho el algoritmo
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, s=5, alpha=0.1)
plt.title("Realidad (PCA)")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_gmm, palette='viridis', s=5, alpha=0.1)
plt.title("Clusterización GMM (Elipses)")

plt.savefig("9_resultado_gmm.png")
print(">> Gráfico guardado: '9_resultado_gmm.png'")