# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:33:31 2026

@author: David
"""

# -*- coding: utf-8 -*-
"""
TFG FASE 2: Ingeniería de Características y Análisis Balanceado
Autor: David Díez Roshan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Configuración estética con fuentes más grandes por defecto
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

# --- 1. CARGA Y LIMPIEZA INICIAL ---
print("\n=== 1. PREPARACIÓN DE DATOS ===")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Limpiamos outliers extremos
datos = datos[(datos['u'] > -100) & (datos['z'] > -100) & (datos['redshift'] >= 0)]

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (EL VECTOR x') ---
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']
print("Nuevas variables generadas (x'): u-g, g-r, r-i, i-z")

# --- 3. EQUILIBRADO DE LA MUESTRA (REMUESTREO) ---
minimo_clase = datos['class'].value_counts().min() 
print(f"Reduciendo todas las clases a {minimo_clase} muestras para equilibrar...")

def equilibrar_datos(df):
    df_bal = df.groupby('class').apply(lambda x: x.sample(minimo_clase, random_state=42))
    return df_bal.reset_index(drop=True)

datos_bal = equilibrar_datos(datos)

# --- 4. VISUALIZACIÓN: DIAGRAMA COLOR-COLOR ---
print("\n=== GENERANDO DIAGRAMAS COLOR-COLOR ===")

# Aumentamos el tamaño y activamos alta resolución (dpi=300)
plt.figure(figsize=(14, 6), dpi=300)

# Gráfica A: Magnitudes Crudas (u vs g)
plt.subplot(1, 2, 1)
sns.scatterplot(data=datos_bal.sample(5000), x='u', y='g', hue='class', s=20, alpha=0.6)
plt.title("Espacio Original (x): Magnitud u vs g", fontsize=16, pad=15)
plt.xlabel("Magnitud u", fontsize=14)
plt.ylabel("Magnitud g", fontsize=14)

# Gráfica B: Colores (g-r vs u-g) -> EJES CORREGIDOS PARA COHERENCIA
plt.subplot(1, 2, 2)
sns.scatterplot(data=datos_bal.sample(5000), x='g-r', y='u-g', hue='class', s=20, alpha=0.6)
plt.title("Espacio Transformado (x'): Color g-r vs u-g", fontsize=16, pad=15)
plt.xlabel("Índice de Color (g - r)", fontsize=14)
plt.ylabel("Índice de Color (u - g)", fontsize=14)
# Ajustamos los límites a la nueva orientación
plt.xlim(-0.5, 2.5) 
plt.ylim(-0.5, 3.5)

plt.tight_layout()
# GUARDADO EN JPG
plt.savefig("4_comparativa_espacios.jpg", format='jpg', bbox_inches='tight')
print(">> Gráfico guardado: '4_comparativa_espacios.jpg'") 

# --- 5. ENTRENAMIENTO PARA EL SUPER PANEL ---
features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos_bal[features]
y = datos_bal['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_sc, y_train)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_sc, y_train)

# --- 6. GENERACIÓN DEL "SÚPER PANEL" (4 VENTANAS) ---
print("\n=== GENERANDO SÚPER PANEL DE ERRORES ===")

def preparar_datos_plot(modelo, X_data, y_real, nombre_set):
    y_pred = modelo.predict(X_data)
    df_plot = pd.DataFrame(scaler.inverse_transform(X_data), columns=features)
    df_plot['Realidad'] = y_real.values
    df_plot['Acierto'] = (y_real.values == y_pred) 
    df_plot['Set'] = nombre_set
    return df_plot

df_rf_train = preparar_datos_plot(rf, X_train_sc, y_train, "RF - Train")
df_rf_test  = preparar_datos_plot(rf, X_test_sc,  y_test,  "RF - Test")
df_svm_train = preparar_datos_plot(svm, X_train_sc, y_train, "SVM - Train")
df_svm_test  = preparar_datos_plot(svm, X_test_sc,  y_test,  "SVM - Test")

# Alta resolución para el super panel
fig, axs = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
markers = {'GALAXY': 'o', 'STAR': '^', 'QSO': 's'} 

lista_plots = [
    (df_rf_train, axs[0, 0], "RF Train"),
    (df_svm_train, axs[0, 1], "SVM Train"),
    (df_rf_test, axs[1, 0], "RF Test"),
    (df_svm_test, axs[1, 1], "SVM Test")
]

for df, ax, titulo in lista_plots:
    df_s = df.sample(min(2000, len(df)), random_state=42)
    
    for clase in ['GALAXY', 'STAR', 'QSO']:
        subset = df_s[df_s['Realidad'] == clase]
        correctos = subset[subset['Acierto'] == True]
        
        # EJES CORREGIDOS: x='g-r', y='u-g'
        ax.scatter(correctos['g-r'], correctos['u-g'], c='blue', marker=markers[clase], label=f'{clase} Ok' if len(correctos)>0 else "", alpha=0.3, s=30)
        
        fallos = subset[subset['Acierto'] == False]
        ax.scatter(fallos['g-r'], fallos['u-g'], c='red', marker=markers[clase], label=f'{clase} Fail' if len(fallos)>0 else "", alpha=0.9, s=60, edgecolors='black', linewidths=0.5)

    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Índice de Color (g - r)", fontsize=14)
    ax.set_ylabel("Índice de Color (u - g)", fontsize=14)
    # Límites adaptados a la nueva orientación
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, linestyle='--', alpha=0.6)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=14, markerscale=1.5)

plt.tight_layout()
# GUARDADO EN JPG
plt.savefig("5_super_panel_errores.jpg", format='jpg', bbox_inches='tight')
print(">> Gráfico guardado: '5_super_panel_errores.jpg'")