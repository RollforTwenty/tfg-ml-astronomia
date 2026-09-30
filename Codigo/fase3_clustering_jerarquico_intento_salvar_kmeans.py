# -*- coding: utf-8 -*-
"""
=============================================================================
TFG FASE 3 - Clustering Jerárquico (Bisecting Strategy)
Un ultimo intento de salvar el K-Means dividiendo en dos pasos.
ESTADO: Experimento Metodológico (Precursor de la Fase 4)
CORRESPONDENCIA: Fase 2 - Clustering Avanzado
=============================================================================

DESCRIPCIÓN:
Este script documenta un intento avanzado de salvar la metodología no 
supervisada mediante una estrategia de 'Bisecting K-Means' (Clustering 
Jerárquico Divisivo). La hipótesis era que una división en cascada podría 
aislar el 'Stellar Locus' del ruido galáctico de fondo de forma más eficiente 
que un K-Means plano.

PROCEDIMIENTO:
1. División binaria inicial (k=2) para segregar la masa principal de galaxias.
2. Re-segmentación del cluster minoritario (k=2) buscando la bimodalidad 
   Estrella-Quásar en el espacio de color u-g.
3. Evaluación mediante tablas de contingencia de la pureza de los sub-clusters.

CONCLUSIÓN TÉCNICA:
A pesar de la sofisticación del proceso, la ausencia de etiquetas de 
entrenamiento impide que el algoritmo resuelva la degeneración fotométrica 
en la zona de solapamiento. Este experimento fue fundamental para decidir 
la arquitectura del 'Clasificador en Cascada' supervisado implementado 
finalmente en la Fase 4.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
# ... (tu código)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. CARGA
print("--- Probando Clustering Jerárquico... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# Feature Engineering (Solo colores, SIN REDSHIFT para ser legales)
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

features = ['u-g', 'g-r', 'r-i', 'i-z'] # Sin redshift
X = datos[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PASO 1: PRIMERA DIVISIÓN (k=2) ---
print("1. Ejecutando K-Means Global (k=2)...")
km1 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels1 = km1.fit_predict(X_scaled)
datos['Nivel1'] = labels1

# Vamos a ver qué ha capturado cada cluster
print(pd.crosstab(datos['class'], datos['Nivel1'], margins=True))

# Identificamos el cluster mayoritario (probablemente Galaxias mezcladas)
counts = datos['Nivel1'].value_counts()
cluster_grande = counts.idxmax()
cluster_peque = counts.idxmin()

print(f"\n--> El Cluster {cluster_peque} parece el interesante. Vamos a re-dividirlo.")

# --- PASO 2: SUB-DIVISIÓN (k=2 en el cluster pequeño) ---
mask_c2 = (datos['Nivel1'] == cluster_peque)
X_sub = X_scaled[mask_c2]
indices_sub = datos[mask_c2].index

if len(X_sub) > 0:
    print("2. Ejecutando K-Means Secundario (k=2)...")
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels2 = km2.fit_predict(X_sub)
    
    # Guardamos resultado
    datos.loc[indices_sub, 'Nivel2'] = labels2
else:
    print("Error: El cluster pequeño está vacío.")

# --- VISUALIZACIÓN FINAL ---
# Asignamos etiquetas finales:
# 0 = Cluster Grande (Nivel 1)
# 1 = Cluster Pequeño A (Nivel 2)
# 2 = Cluster Pequeño B (Nivel 2)

datos['Final_Cluster'] = 0 # Por defecto el grande
datos.loc[mask_c2, 'Final_Cluster'] = datos.loc[mask_c2, 'Nivel2'] + 1 # 1 y 2

print("\n=== TABLA FINAL CLUSTERING JERÁRQUICO ===")
print(pd.crosstab(datos['class'], datos['Final_Cluster'], 
                  colnames=['Cluster Final (0=Fondo, 1=SubA, 2=SubB)']))

# Gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(data=datos.sample(5000), x='u-g', y='g-r', hue='Final_Cluster', palette='viridis', s=10)
plt.title("Bisecting K-Means: ¿Mejora la separación?")
plt.savefig("16_clustering_jerarquico.png")
print(">> Gráfico guardado.")