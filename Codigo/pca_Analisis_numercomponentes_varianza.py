# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 02:51:18 2025

@author: David
"""

# --- ANÁLISIS DE VARIANZA PCA ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Cargar y Preparar (igual que antes)
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
# Limpieza básica
datos = datos[(datos['u'] > -100) & (datos['z'] > -100) & (datos['redshift'] >= 0)]

# 2. Ingeniería de características (x') + Redshift
# Usamos las mismas variables que dieron buen resultado antes
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos[features]

# 3. Escalado (OBLIGATORIO para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA Completo (pedimos todas las componentes posibles, que son 5)
pca = PCA(n_components=None) 
pca.fit(X_scaled)

# 5. Obtener la Varianza Explicada
varianza = pca.explained_variance_ratio_
varianza_acumulada = pca.explained_variance_ratio_.cumsum()

print("\n--- RESULTADOS PCA ---")
for i, v in enumerate(varianza_acumulada):
    print(f"Con {i+1} componentes explicamos el {v*100:.2f}% de la varianza.")

# 6. Gráfico de Codo (Scree Plot)
plt.figure(figsize=(10, 6))
x_axis = range(1, len(features) + 1)

# Barras: Varianza individual
plt.bar(x_axis, varianza, alpha=0.6, label='Varianza Individual', color='teal')
# Línea: Varianza acumulada
plt.plot(x_axis, varianza_acumulada, marker='o', linestyle='-', color='red', label='Varianza Acumulada')

# Línea de corte al 95% (típico criterio estadístico)
plt.axhline(y=0.95, color='gray', linestyle='--', label='Umbral 95%')

plt.xlabel('Número de Componentes Principales')
plt.ylabel('Ratio de Varianza Explicada')
plt.title('Análisis de Varianza PCA (Scree Plot)')
plt.xticks(x_axis)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("6_pca_varianza.png")
print(">> Gráfico guardado: '6_pca_varianza.png'")

print(pca.feature_names_in_)
