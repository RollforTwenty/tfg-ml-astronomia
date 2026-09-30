# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:34:25 2026

@author: David
"""
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# 1. Cargar los datos crudos de Kaggle
df = pd.read_csv("star_classification.csv")

# 2. Ingeniería de características (¡lo que justificas en el Capítulo 3!)
df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']

# 3. Definir predictores (X) y variable objetivo (y)
X_ejemplo = df[['u-g', 'g-r']] 
y_ejemplo = df['class']

# 4. Entrenamos un árbol muy básico (max_depth=2)
arbol_didactico = DecisionTreeClassifier(max_depth=2, random_state=42, criterion='gini')
arbol_didactico.fit(X_ejemplo, y_ejemplo)

# 5. Configuramos el lienzo en alta resolución para LaTeX
plt.figure(figsize=(10, 6), dpi=300)

# 6. Dibujamos el árbol
plot_tree(arbol_didactico, 
          feature_names=['u-g', 'g-r'], 
          class_names=list(arbol_didactico.classes_), # Pilla los nombres automáticamente
          filled=True,      
          rounded=True,     
          fontsize=12,
          proportion=False) 

plt.title("Ejemplo Didáctico: Partición del Espacio Fotométrico", fontsize=14, pad=20)
plt.tight_layout()

# 7. Guardamos la imagen
plt.savefig("arbol_decision_ejemplo.jpg", format="jpg", bbox_inches='tight', dpi=300)
plt.show()