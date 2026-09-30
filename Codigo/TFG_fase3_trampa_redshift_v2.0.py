# -*- coding: utf-8 -*-
"""
=============================================================================
FASE DEL TFG: Fase 3 - Experimento de Control (Limitaciones No Supervisadas)
Objetivo: Demostrar que K-Means 'funciona' solo porque corta el eje Redshift.
=============================================================================

DESCRIPCIÓN:
Este script documenta el experimento de control diseñado para demostrar las 
limitaciones topológicas intrínsecas de los algoritmos de agrupamiento 
espacial (K-Means) aplicados a catálogos fotométricos puramente dependientes 
del color.

PROCEDIMIENTO:
1. Se incorpora la variable de distancia cosmológica (redshift) al vector 
   de características de entrenamiento.
2. Tras el escalado estándar, se ajusta el algoritmo K-Means (k=3).
3. Se genera un panel comparativo bidimensional (Color u-g vs Redshift) 
   contrastando las etiquetas reales (SDSS) con las agrupaciones inferidas.

CONCLUSIÓN ASTROFÍSICA Y ALGORÍTMICA:
El experimento corrobora empíricamente que el redshift actúa como una variable 
de confusión dominante. Al incluirlo, K-Means segmenta el espacio basándose 
casi exclusivamente en cortes ortogonales sobre el eje de distancia (estrellas 
en z~0, galaxias en z bajo, QSO en z alto), ignorando la varianza fotométrica. 
Esto demuestra que, en ausencia de espectroscopía (redshift), el 'Stellar Locus' 
y los cuásares forman un continuo topológico inseparable para K-Means, 
justificando el salto hacia arquitecturas supervisadas (Fase 4).
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuración global para fuentes grandes ---
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

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
# Convertimos las etiquetas numéricas a texto para la leyenda (Cluster 1, 2, 3)
datos['Cluster_KMeans'] = [f"Cluster {k+1}" for k in labels]
datos['u-g'] = datos['u'] - datos['g'] # Calculamos color para el eje X

# Tomamos una muestra fija para no saturar y acelerar el renderizado
datos_sample = datos.sample(5000, random_state=42)

# Unificamos colores de la realidad con el resto de las figuras del TFG
colores_reales = {'GALAXY': 'gray', 'STAR': 'blue', 'QSO': 'green'}

plt.figure(figsize=(14, 7), dpi=300)

# --- Gráfico 1: La Realidad (Coloreado por Clase Real) ---
plt.subplot(1, 2, 1)
sns.scatterplot(data=datos_sample, x='u-g', y='redshift', hue='class', 
                palette=colores_reales, s=50, alpha=0.6, edgecolor='black', linewidth=0.5)

plt.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label="Límite Estelar")

plt.title("REALIDAD: Color vs Redshift", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Índice de Color (u - g)", fontsize=14, fontweight='bold')
plt.ylabel("Redshift (z)", fontsize=14, fontweight='bold')
plt.ylim(-0.5, 4)

plt.legend(title="Clase Taxonómica", title_fontsize='13', fontsize='12', frameon=True, shadow=True)

# --- Gráfico 2: Lo que ve K-Means (Coloreado por Cluster) ---
plt.subplot(1, 2, 2)
sns.scatterplot(data=datos_sample, x='u-g', y='redshift', hue='Cluster_KMeans', 
                palette='Set1', s=50, alpha=0.6, edgecolor='black', linewidth=0.5,
                hue_order=['Cluster 1', 'Cluster 2', 'Cluster 3'])

plt.title("K-MEANS: Separación inducida por Redshift", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Índice de Color (u - g)", fontsize=14, fontweight='bold')
plt.ylabel("Redshift (z)", fontsize=14, fontweight='bold')
plt.ylim(-0.5, 4)

plt.legend(title="Agrupación K-Means", title_fontsize='13', fontsize='12', frameon=True, shadow=True)

# Guardado en alta calidad
plt.tight_layout()
plt.savefig("14_trampa_redshift.jpg", format='jpg', bbox_inches='tight', dpi=300)
print(">> Gráfico guardado: '14_trampa_redshift.jpg'")

# Tabla de contingencia para confirmar
print("\nTABLA DE CONTINGENCIA (K-Means con Redshift):")
print(pd.crosstab(y, labels, rownames=['Realidad'], colnames=['Cluster KMeans']))