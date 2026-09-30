# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:12:00 2026
@author: David
Reducción de Dimensionalidad (PCA) - Análisis de Varianza
Autor: David Díez Roshan

=============================================================================
ARCHIVO: 02_pca_analisis_varianza.py
FASE DEL TFG: Fase 2 - Investigación de Métodos No Supervisados
=============================================================================

DESCRIPCIÓN:
Este script ejecuta un Análisis de Componentes Principales (PCA) completo 
sobre el espacio de características estandarizado (índices de color + redshift). 
Su objetivo principal no es la clasificación, sino el estudio algebraico de 
la varianza del conjunto de datos mediante la extracción de los autovalores 
de la matriz de covarianza.

PROCEDIMIENTO:
1. Ingeniería de características y escalado estándar (StandardScaler).
2. Ajuste del modelo PCA sin restricción de componentes (n_components=None).
3. Cálculo de la varianza explicada individual y acumulada.
4. Generación del gráfico de codo (Scree Plot) para evaluar visualmente 
   el punto óptimo de corte topológico (criterio empírico del 95% de varianza).

CONCLUSIÓN MATEMÁTICA:
El gráfico resultante documenta matemáticamente la complejidad intrínseca 
del conjunto de datos, justificando las decisiones de reducción de 
dimensionalidad tomadas para las posteriores fases de visualización y 
agrupamiento no supervisado.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración estética profesional con fuentes grandes
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

# 1. Cargar y Preparar
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
# Limpieza básica
datos = datos[(datos['u'] > -100) & (datos['z'] > -100) & (datos['redshift'] >= 0)]

# 2. Ingeniería de características (x') + Redshift
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

# 6. Gráfico de Codo (Scree Plot) - ALTA RESOLUCIÓN
plt.figure(figsize=(10, 6), dpi=300)
x_axis = range(1, len(features) + 1)

# Barras: Varianza individual
plt.bar(x_axis, varianza, alpha=0.6, label='Varianza Individual', color='teal')
# Línea: Varianza acumulada (ahora más gruesa y con marcadores más grandes)
plt.plot(x_axis, varianza_acumulada, marker='o', markersize=8, linestyle='-', linewidth=2, color='red', label='Varianza Acumulada')

# Línea de corte al 95% (típico criterio estadístico)
plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, label='Umbral 95%')

# Textos grandes y legibles
plt.xlabel('Número de Componentes Principales', fontsize=14, fontweight='bold')
plt.ylabel('Ratio de Varianza Explicada', fontsize=14, fontweight='bold')
plt.title('Análisis de Varianza PCA (Scree Plot)', fontsize=16, fontweight='bold', pad=15)
plt.xticks(x_axis, fontsize=12)
plt.yticks(fontsize=12)

# Leyenda con un recuadro claro
plt.legend(fontsize=12, loc='center right', frameon=True, shadow=True)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Guardado en JPG con calidad de imprenta
plt.tight_layout()
plt.savefig("6_pca_varianza.jpg", format='jpg', bbox_inches='tight', dpi=300)
print(">> Gráfico guardado: '6_pca_varianza.jpg'")

print("\nVariables originales introducidas al modelo:", features)