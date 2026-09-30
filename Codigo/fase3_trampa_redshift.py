
# -*- coding: utf-8 -*-
"""
TFG FASE 3 - PASO FINAL: La "Trampa" del Redshift
Objetivo: Demostrar que K-Means 'funciona' solo porque corta el eje Redshift.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. CARGA
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. EXPERIMENTO: K-MEANS CON REDSHIFT DOMINANTE
# Incluimos 'redshift' en las variables para ver cómo el algoritmo se vicia con él
features = ['u', 'g', 'r', 'i', 'z', 'redshift']
X = datos[features]
y = datos['class']

# Escalamos (El redshift al escalarse se vuelve muy discriminante)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Ejecutando K-Means (k=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 3. VISUALIZACIÓN REVELADORA
datos['Cluster_KMeans'] = labels
datos['u-g'] = datos['u'] - datos['g'] # Calculamos color para el eje X

plt.figure(figsize=(12, 6))

# Gráfico 1: La Realidad (Coloreado por Clase Real)
plt.subplot(1, 2, 1)
sns.scatterplot(data=datos.sample(5000), x='u-g', y='redshift', hue='class', palette='bright', s=10, alpha=0.6)
plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.3, label="Límite Estelar")
plt.title("REALIDAD: Color vs Redshift")
plt.ylim(-0.5, 4)

# Gráfico 2: Lo que ve K-Means (Coloreado por Cluster)
plt.subplot(1, 2, 2)
sns.scatterplot(data=datos.sample(5000), x='u-g', y='redshift', hue='Cluster_KMeans', palette='viridis', s=10, alpha=0.6)
plt.title("K-MEANS: ¿Física o Cortes en Redshift?")
plt.ylim(-0.5, 4)

plt.tight_layout()
plt.savefig("14_trampa_redshift.png")
print(">> Gráfico guardado: '14_trampa_redshift.png'")

# Tabla de contingencia para confirmar
print("\nTABLA DE CONTINGENCIA (K-Means con Redshift):")
print(pd.crosstab(y, labels, rownames=['Realidad'], colnames=['Cluster KMeans']))