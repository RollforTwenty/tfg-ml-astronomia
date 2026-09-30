# -*- coding: utf-8 -*-
"""

=============================================================================
TFG FASE 3: Exploración Visual de Combinaciones Lineales
Objetivo: Buscar combinaciones de colores (u-g, g-r, etc.) que separen visualmente las clases.
ESTADO: Obsoleto / Versión Preliminar
CORRESPONDENCIA: Capítulo 3 - Análisis Exploratorio (Borrador)
=============================================================================

DESCRIPCIÓN:
Primer intento de visualización masiva de variables mediante Pairplot. 

MOTIVO DE OBSOLESCENCIA:
El uso de un muestreo aleatorio simple (sample de 5000 puntos totales) genera 
una representación visual sesgada debido al desbalanceo de clases del SDSS. 
La abundancia de galaxias oculta la topología de los cuásares y estrellas. 
Este script fue sustituido por la versión de 'muestreo equilibrado' 
(01b_eda_matriz_dispersion_colores.py) para un análisis científico riguroso.
=============================================================================
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CARGA DE DATOS
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Limpieza básica de errores de medición (-9999, etc.) para que no rompan la escala del gráfico
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. CREACIÓN DE LAS 4 COMBINACIONES LINEALES (Colores)
# Estas son las restas que eliminan la distancia
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

# Seleccionamos solo las columnas nuevas y la clase
cols_analisis = ['u-g', 'g-r', 'r-i', 'i-z', 'class']
df_plot = datos[cols_analisis]

# 3. TOMAR UNA MUESTRA (Para que el gráfico sea rápido y legible)
# Con 100.000 puntos es una mancha. Con 5.000 vemos la estructura.
df_sample = df_plot.sample(5000, random_state=42)

print("--- Generando Matriz de Dispersión (Pairplot)... ---")

# 4. GENERAR EL GRÁFICO "TODOS CONTRA TODOS"
# Esto crea una rejilla comparando cada variable con todas las demás
plt.figure(figsize=(15, 15))
sns.pairplot(df_sample, hue='class', 
             palette='bright',  # Usamos colores vivos para distinguir bien
             plot_kws={'alpha': 0.6, 's': 15}, # Transparencia y tamaño de punto
             diag_kind='kde') # En la diagonal ponemos curvas de densidad

plt.savefig("7_exploracion_lineales.png")
print(">> Gráfico guardado: '7_exploracion_lineales.png'")
print("Revisa la imagen. Busca pares de variables donde los grupos no se solapen.")