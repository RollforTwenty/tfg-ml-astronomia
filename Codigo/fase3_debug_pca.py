# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:03:18 2025
@author: David

=============================================================================
ESTADO: Script de Depuración / Histórico
CORRESPONDENCIA: Fase 2 y Fase 4 (Ajustes Técnicos)
=============================================================================

DESCRIPCIÓN:
Este script documenta varias pruebas de control de calidad realizadas durante 
el desarrollo del TFG. No genera resultados para la memoria, sino que sirve 
como herramienta de diagnóstico para el investigador.

HITOS TÉCNICOS:
1. Análisis de Pesos (Loadings) del PCA: Se verifica matemáticamente que las 
   componentes principales guardan una relación lineal con los índices de 
   color físicos (u-g, g-r, etc.).
2. Optimización de Visualización de Errores: Implementación de una estrategia 
   de sobreposición de capas (puntos rojos sobre azules) para evitar el 
   sesgo visual por saturación de datos (overplotting) en los gráficos 
   de diagnóstico del Random Forest.

CONCLUSIÓN:
Este archivo permitió refinar la técnica de trazado utilizada posteriormente 
en el 'Súper Panel' definitivo (Figura 4.1), garantizando que los fallos de 
clasificación fueran siempre visibles para el análisis estadístico.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Carga
datos = pd.read_csv("star_classification.csv")
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# --- PUNTO 2: ZOOM r-i vs i-z ---
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']
plt.figure(figsize=(8, 6))
sns.scatterplot(data=datos.sample(5000), x='r-i', y='i-z', hue='class', s=10, alpha=0.5)
plt.title("Zoom Fotométrico: r-i vs i-z")
plt.savefig("11_zoom_ri_iz.png")

# --- PUNTO 4: PCA A CAPÓN (Crudo) ---
# Metemos u,g,r,i,z crudos
X_raw = datos[['u', 'g', 'r', 'i', 'z']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca_raw = PCA(n_components=2)
pca_raw.fit(X_scaled)

# Miramos los pesos (loadings) de cada variable en el PCA
loadings = pd.DataFrame(pca_raw.components_.T, columns=['PC1', 'PC2'], index=['u', 'g', 'r', 'i', 'z'])
print("\n=== PESOS DEL PCA (CARACTERÍSTICAS CRUDAS) ===")
print(loadings)
print("\nSi PC1 tiene valores opuestos en u y g (ej: 0.5 y -0.5), significa que PCA está 'restando' esas variables (u-g).")

# --- PUNTO 3: DEBUG PUNTOS ROJOS ---
# (Modificamos el orden de pintado para que el ROJO esté siempre arriba)
# Repetimos el entrenamiento de RF rápido
X_train, X_test, y_train, y_test = train_test_split(X_scaled, datos['class'], test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

# Pintamos el gráfico de RF Train forzando la visualización de errores
y_pred_train = rf.predict(X_train)
correctos = (y_train == y_pred_train)

# Proyectamos para pintar
X_train_pca = pca_raw.transform(X_train)

plt.figure(figsize=(8, 6))
# Pintamos primero los bien clasificados (azules)
plt.scatter(X_train_pca[correctos, 0], X_train_pca[correctos, 1], c='blue', alpha=0.2, s=5, label='Ok')
# Pintamos los MAL clasificados (rojos) AL FINAL y MÁS GRANDES
plt.scatter(X_train_pca[~correctos, 0], X_train_pca[~correctos, 1], c='red', alpha=0.8, s=20, label='Error')
plt.title("Debug Errores RF Train (Puntos rojos forzados)")
plt.legend()
plt.savefig("12_debug_rf_train_errores.png")
