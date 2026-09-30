# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 04_exploracion_densidad_dbscan.py
FASE DEL TFG: Fase 2 - Clustering por Densidad Espacial.
Objetivo: Superar la limitación de esfericidad de K-Means. Clustering por Densidad (DBSCAN)
=============================================================================

DESCRIPCIÓN:
Este script documenta la última iteración dentro de la experimentación con
algoritmos no supervisados. Ante las limitaciones geométricas impuestas por 
K-Means (esfericidad) y GMM (elipticidad), se implementa DBSCAN, un algoritmo 
basado puramente en la densidad topológica que permite identificar clústeres 
de formas no convexas y aislar ruido (outliers fotométricos).

PROCEDIMIENTO:
1. Generación de las características fotométricas (índices de color).
2. Escalado estricto de los datos (StandardScaler). Este paso es matemáticamente 
   crítico para DBSCAN, ya que el cálculo de la vecindad (parámetro epsilon) 
   requiere que todas las dimensiones (color y redshift) operen en la misma 
   escala métrica.
3. Configuración empírica de DBSCAN con eps=0.5 y min_samples=50.
4. Generación de proyecciones espaciales mediante PCA para evaluar visualmente 
   la segmentación del continuo térmico.

CONCLUSIÓN EXPERIMENTAL:
A pesar de su capacidad para modelar fronteras no lineales, el análisis 
demostró que la agrupación por densidad tiende a fusionar la población 
estelar y extragaláctica en un único macro-clúster si no se segmenta el 
espacio mediante algoritmos supervisados de partición ortogonal, lo que 
motivó el avance hacia la Fase 4 (Random Forest + SMOTE).
=============================================================================
"""


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. CARGA
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. SELECCIÓN DE VARIABLES (Colores + Redshift)
# Usamos lo que tiene más física: Colores
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos[features]
y = datos['class']

# 3. ESCALADO (CRÍTICO para DBSCAN)
# DBSCAN mide distancias, si no escalas, el redshift (0-7) pesará distinto que el color (0-2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. EJECUTAR DBSCAN
# eps: Radio de búsqueda (la "distancia" máxima para ser vecino)
# min_samples: Cuántos puntos hacen falta para formar un grupo denso
print("Ejecutando DBSCAN (esto puede tardar un poco)...")
# Nota: eps=0.5 es un valor estándar inicial para datos escalados (normalizados)
dbscan = DBSCAN(eps=0.5, min_samples=50)
labels_db = dbscan.fit_predict(X_scaled)

# El -1 significa "RUIDO" (puntos que no pertenecen a ningún cluster)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_ruido = list(labels_db).count(-1)

print(f"\nResultados DBSCAN:")
print(f"Clústeres encontrados: {n_clusters}")
print(f"Puntos clasificados como Ruido (-1): {n_ruido} (de {len(datos)})")

# 5. TABLA DE CONTINGENCIA
print("\n=== TABLA DE CONTINGENCIA (DBSCAN) ===")
print(pd.crosstab(y, labels_db, rownames=['Realidad'], colnames=['Cluster DBSCAN']))

# 6. VISUALIZACIÓN PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Realidad
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, s=10, alpha=0.3)
plt.title("Realidad")

# DBSCAN
plt.subplot(1, 2, 2)
# Pintamos el ruido (-1) en gris y flojito
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_db, palette='tab10', s=10, alpha=0.5)
plt.title(f"DBSCAN ({n_clusters} clusters + Ruido)")

plt.savefig("13_resultado_dbscan.png")
print(">> Gráfico guardado: '13_resultado_dbscan.png'")