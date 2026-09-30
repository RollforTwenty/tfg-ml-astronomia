# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 01_visualizacion_stellar_locus.py
FASE DEL TFG: Capítulo 3 - Ingeniería de Características (Análisis Físico)
=============================================================================

DESCRIPCIÓN:
Este script tiene un propósito puramente astrofísico y exploratorio. Sirve 
para justificar visual y físicamente la transformación de los datos crudos 
(magnitudes aparentes) en un espacio de características basado en índices 
de color termodinámicos.

PROCEDIMIENTO:
1. Se calculan los índices de color clave: u-g y g-r.
2. Se aísla una muestra estratificada que contiene exclusivamente la clase 
   estelar (STAR) y los núcleos galácticos activos (QSO).
3. Se proyectan ambas poblaciones en el plano color-color para analizar 
   su separabilidad topológica bidimensional.

CONCLUSIÓN ASTROFÍSICA:
La gráfica resultante demuestra visualmente el fenómeno del 'exceso UV'. 
Mientras que las estrellas conforman una secuencia térmica bien definida 
(Stellar Locus), los cuásares presentan valores fuertemente negativos en 
el índice u-g debido a la emisión no térmica de sus discos de acreción. 
Esto valida la elección de los filtros u y g como los descriptores de 
características más potentes para la separación de estas dos clases.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CARGAMOS LOS DATOS ---
# OJO: Poner aquí la ruta real o el nombre exacto de mi archivo CSV de Kaggle
df = pd.read_csv('star_classification.csv') 

# --- 2. CALCULAMOS LOS COLORES ---
# Al cargar el CSV crudo, necesitamos calcular las restas matemáticas
df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']

# --- 3. FILTRAMOS Y CREAMOS LA MUESTRA ---
df_plot = df[df['class'].isin(['STAR', 'QSO'])]

df_sample = pd.concat([
    df_plot[df_plot['class'] == 'STAR'].sample(n=3000, random_state=42),
    df_plot[df_plot['class'] == 'QSO'].sample(n=3000, random_state=42)
])

# --- 4. CONFIGURAMOS Y DIBUJAMOS LA GRÁFICA ---
plt.figure(figsize=(8, 6), dpi=300)

sns.scatterplot(
    data=df_sample, 
    x='g-r', 
    y='u-g', 
    hue='class', 
    palette={'STAR': '#1f77b4', 'QSO': '#d62728'}, 
    s=15,          
    alpha=0.5,     
    edgecolor=None
)

plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 3.5)

plt.title("Diagrama Color-Color: Separación Termodinámica", fontsize=14, pad=15)
plt.xlabel("Índice de Color ($g - r$)", fontsize=12)
plt.ylabel("Índice de Color ($u - g$)", fontsize=12)

# --- 5. ANOTACIONES FÍSICAS ---
plt.annotate('Stellar Locus\n(Secuencia Térmica)', 
             xy=(1.2, 1.3), xytext=(1.6, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.annotate('Cuásares (QSO)\n(Exceso UV)', 
             xy=(0.2, 0.3), xytext=(-0.1, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.legend(title='Clase Espectral', loc='upper right', framealpha=1)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("stellar_locus_quasars.jpg", format="jpg", bbox_inches='tight', dpi=300)
plt.show()