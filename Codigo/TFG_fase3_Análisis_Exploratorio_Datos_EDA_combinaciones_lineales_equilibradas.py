# -*- coding: utf-8 -*-
"""
=============================================================================

FASE DEL TFG: Capítulo 3 - Análisis Exploratorio de Datos (EDA)
Exploración Visual de Combinaciones Lineales (CORREGIDO)
Objetivo: Buscar combinaciones de colores con MUESTRA EQUILIBRADA.
=============================================================================

DESCRIPCIÓN:
Este script documenta la exploración topológica inicial del espacio de 
características. Genera una matriz de dispersión (Pairplot) que proyecta 
ortogonalmente todas las combinaciones posibles de los índices de color 
termodinámicos calculados.

PROCEDIMIENTO:
1. Cálculo de los cuatro índices de color estándar (u-g, g-r, r-i, i-z).
2. Submuestreo estratificado estricto (1000 observaciones por clase). 
   NOTA METODOLÓGICA: A diferencia de la Fase de Modelado (donde se emplea 
   SMOTE), en esta etapa de visualización exploratoria se utiliza el 
   submuestreo (Under-sampling) para evitar el sesgo óptico ('overplotting') 
   generado por la clase mayoritaria (Galaxias), permitiendo analizar la 
   distribución real de las tres topologías.
3. Generación de la cuadrícula de dispersión cruzada con estimación de 
   densidad del núcleo (KDE) en la diagonal principal.

CONCLUSIÓN:
Esta matriz permite evaluar visualmente qué combinaciones lineales de 
características ofrecen la mayor entropía y capacidad discriminante, guiando 
la selección de variables para los posteriores modelos de Machine Learning.
=============================================================================
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CARGA DE DATOS
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Limpieza básica
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. CREACIÓN DE LAS 4 COMBINACIONES LINEALES
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

cols_analisis = ['u-g', 'g-r', 'r-i', 'i-z', 'class']
df_completo = datos[cols_analisis]

# 3. EQUILIBRADO DE LA MUESTRA (¡EL PASO CLAVE!)
# Queremos ver las clases por igual, así que cogemos 1000 de cada una para el gráfico
muestras_por_clase = 1000 
print(f"--- Tomando {muestras_por_clase} puntos de cada clase para visualizar bien... ---")

df_balanced = df_completo.groupby('class').apply(
    lambda x: x.sample(n=min(len(x), muestras_por_clase), random_state=42)
).reset_index(drop=True)

# 4. GENERAR EL GRÁFICO "TODOS CONTRA TODOS"
print("--- Generando Matriz de Dispersión... ---")
plt.figure(figsize=(15, 15))
sns.pairplot(df_balanced, hue='class', 
             palette='bright', 
             plot_kws={'alpha': 0.5, 's': 20}, # Puntos un poco más grandes
             diag_kind='kde') 

plt.suptitle("Exploración de Colores (Muestra Equilibrada)", y=1.02)
plt.savefig("7_exploracion_lineales_v2.png")
print(">> Gráfico guardado: '7_exploracion_lineales_v2.png'")