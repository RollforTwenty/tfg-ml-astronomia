# -*- coding: utf-8 -*-
"""

=============================================================================
ARCHIVO: 02_exploracion_bandas_infrarrojas.py
ESTADO: Experimento Exploratorio (Descartado para modelado final)
CORRESPONDENCIA: Capítulo 3 - Ingeniería de Características
TFG FASE 3 - PASO 1: Zoom Fotométrico (r-i vs i-z)
Objetivo: Verificar si las bandas infrarrojas separan mejor las clases.
=============================================================================

DESCRIPCIÓN:
Este script documenta un análisis exploratorio del espacio fotométrico en el 
extremo rojo e infrarrojo cercano del espectro (bandas r, i, z). El objetivo 
era comprobar empíricamente si la combinación de estos índices (r-i vs i-z) 
ofrecía una mejor separabilidad topológica de las clases que las bandas 
ultravioleta-ópticas (u-g, g-r).

CONCLUSIÓN ASTROFÍSICA:
La visualización resultante demostró un solapamiento masivo entre las tres 
clases taxonómicas. Esto corroboró la hipótesis de que el mayor poder 
discriminante reside en el exceso ultravioleta de los cuásares (banda u), 
justificando la exclusión de las proyecciones puramente infrarrojas como 
espacio de decisión principal para el entrenamiento de los modelos 
supervisados en la Fase 4.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGA DE DATOS
print("--- Cargando datos... ---")
nombre_archivo = "star_classification.csv"
# Cargamos saltándonos los IDs corruptos si es necesario, aunque aquí solo pintamos
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Limpieza básica (Quitar errores -9999)
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# 2. INGENIERÍA DE CARACTERÍSTICAS (Solo las que pidió la tutora)
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

# 3. GENERAR EL GRÁFICO
print("Generando gráfico específico r-i vs i-z...")

plt.figure(figsize=(10, 8))
# Tomamos una muestra para que el gráfico no sea una mancha sólida
df_sample = datos.sample(5000, random_state=42)

sns.scatterplot(
    data=df_sample, 
    x='r-i', 
    y='i-z', 
    hue='class', 
    palette='bright', 
    s=15, 
    alpha=0.6
)

plt.title("Espacio de Color Infrarrojo: (r-i) vs (i-z)")
plt.xlabel("Color r-i")
plt.ylabel("Color i-z")
plt.grid(True, linestyle='--', alpha=0.5)

# Guardamos
plt.savefig("11_zoom_ri_iz.png")
print(">> Gráfico guardado: '11_zoom_ri_iz.png'")