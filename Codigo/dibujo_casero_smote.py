# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:18:44 2026

@author: David
"""

import matplotlib.pyplot as plt
import numpy as np

# Configurar gráfico alta resolución
plt.figure(figsize=(8, 6), dpi=300)

# Puntos de la clase mayoritaria (Galaxias - Fondo)
np.random.seed(42)
X_maj = np.random.normal(loc=[5, 5], scale=1.5, size=(40, 2))
plt.scatter(X_maj[:, 0], X_maj[:, 1], color='lightgray', alpha=0.6, s=100, label='Clase Mayoritaria')

# Puntos de la clase minoritaria (Cuásares - Originales)
X_min = np.array([[2, 2], [3, 2.5], [2.5, 3.5], [1.5, 2.8]])
plt.scatter(X_min[:, 0], X_min[:, 1], color='blue', edgecolor='black', s=120, label='Clase Minoritaria (Original)', zorder=3)

# Dibujar las líneas de los k-NN (segmentos)
punto_origen = X_min[1] # Elegimos el punto [3, 2.5]
vecino = X_min[2]       # Su vecino [2.5, 3.5]

plt.plot([punto_origen[0], vecino[0]], [punto_origen[1], vecino[1]], 'k--', zorder=1)

# Generar un punto sintético SMOTE (interpolación)
delta = 0.6 # Un valor aleatorio entre 0 y 1
punto_sintetico = punto_origen + delta * (vecino - punto_origen)

plt.scatter(punto_sintetico[0], punto_sintetico[1], color='red', marker='*', edgecolor='black', s=300, label='Muestra Sintética (SMOTE)', zorder=4)

# Anotaciones matemáticas para que quede súper académico
plt.annotate(r'$\mathbf{x}_i$', xy=(punto_origen[0]+0.1, punto_origen[1]-0.1), fontsize=14)
plt.annotate(r'$\mathbf{x}_{vecino}$', xy=(vecino[0]-0.1, vecino[1]+0.2), fontsize=14)
plt.annotate(r'$\mathbf{x}_{new}$', xy=(punto_sintetico[0]+0.15, punto_sintetico[1]), fontsize=14, color='red')

# Estética
plt.title("Generación de Muestras Sintéticas mediante SMOTE", fontsize=15, pad=15)
plt.xlabel("Característica 1 (ej. Color $u-g$)", fontsize=12)
plt.ylabel("Característica 2 (ej. Color $g-r$)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Guardar
plt.savefig("esquema_smote_propio.jpg", format="jpg", bbox_inches='tight', dpi=300)
plt.show()