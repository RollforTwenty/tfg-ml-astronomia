# -*- coding: utf-8 -*-
"""
TFG FASE 2: Ingeniería de Características, Análisis Balanceado y Súper Panel
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

# --- CONFIGURACIÓN ESTÉTICA PROFESIONAL ---
sns.set_theme(style="whitegrid", rc={
    "axes.labelsize": 14, 
    "xtick.labelsize": 12, 
    "ytick.labelsize": 12,
    "font.weight": "normal"
})

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

# --- 3. EQUILIBRADO DE LA MUESTRA ---
minimo_clase = datos['class'].value_counts().min() 
print(f"Equilibrando clases a {minimo_clase} muestras...")

def equilibrar_datos(df):
    df_bal = df.groupby('class').apply(lambda x: x.sample(minimo_clase, random_state=42))
    return df_bal.reset_index(drop=True)

datos_bal = equilibrar_datos(datos)

# =============================================================================
# --- 4. VISUALIZACIÓN: COMPARATIVA DE ESPACIOS (FIGURA 3.3) ---
# =============================================================================
print("\n=== GENERANDO FIGURA: 4_comparativa_espacios.jpg ===")

plt.figure(figsize=(14, 6))

# Gráfica A: Magnitudes Crudas (u vs g)
plt.subplot(1, 2, 1)
sns.scatterplot(data=datos_bal.sample(5000), x='u', y='g', hue='class', s=20, alpha=0.6)
plt.title("Espacio Original (x): Magnitud u vs g", fontsize=16, pad=15, fontweight='bold')
plt.xlabel("Magnitud u", fontsize=14)
plt.ylabel("Magnitud g", fontsize=14)

# Gráfica B: Colores (g-r vs u-g) -> EJES COHERENTES CON EL RESTO DEL TFG
plt.subplot(1, 2, 2)
sns.scatterplot(data=datos_bal.sample(5000), x='g-r', y='u-g', hue='class', s=20, alpha=0.6)
plt.title("Espacio Transformado (x'): Color g-r vs u-g", fontsize=16, pad=15, fontweight='bold')
plt.xlabel("Índice de Color (g - r)", fontsize=14)
plt.ylabel("Índice de Color (u - g)", fontsize=14)
plt.xlim(-0.5, 2.5) 
plt.ylim(-0.5, 3.5)

plt.tight_layout()
# OPTIMIZACIÓN PARA OVERLEAF
plt.savefig("4_comparativa_espacios.jpg", format='jpg', dpi=150, pil_kwargs={'quality': 80}, bbox_inches='tight')
print(">> Guardado con éxito.") 

# =============================================================================
# --- 5. ENTRENAMIENTO PARA EL SUPER PANEL ---
# =============================================================================
features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos_bal[features]
y = datos_bal['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("\nEntrenando modelos supervisados...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_sc, y_train)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_sc, y_train)

# =============================================================================
# --- 6. GENERACIÓN DEL "SÚPER PANEL" (FIGURA 4.1) ---
# =============================================================================
print("\n=== GENERANDO FIGURA: 5_super_panel_errores.jpg ===")

def preparar_datos_plot(modelo, X_data, y_real, nombre_set):
    y_pred = modelo.predict(X_data)
    df_plot = pd.DataFrame(scaler.inverse_transform(X_data), columns=features)
    df_plot['Realidad'] = y_real.values
    df_plot['Acierto'] = (y_real.values == y_pred) 
    return df_plot

df_rf_train = preparar_datos_plot(rf, X_train_sc, y_train, "RF Train")
df_rf_test  = preparar_datos_plot(rf, X_test_sc,  y_test,  "RF Test")
df_svm_train = preparar_datos_plot(svm, X_train_sc, y_train, "SVM Train")
df_svm_test  = preparar_datos_plot(svm, X_test_sc,  y_test,  "SVM Test")

fig, axs = plt.subplots(2, 2, figsize=(16, 14))
markers = {'GALAXY': 'o', 'STAR': '^', 'QSO': 's'} 
colors_ok = {'GALAXY': 'gray', 'STAR': 'blue', 'QSO': 'green'}

lista_plots = [
    (df_rf_train, axs[0, 0], "Random Forest (Entrenamiento)"),
    (df_svm_train, axs[0, 1], "SVM (Entrenamiento)"),
    (df_rf_test, axs[1, 0], "Random Forest (Prueba)"),
    (df_svm_test, axs[1, 1], "SVM (Prueba)")
]

for df, ax, titulo in lista_plots:
    df_s = df.sample(min(2500, len(df)), random_state=42)
    
    for clase in ['GALAXY', 'STAR', 'QSO']:
        subset = df_s[df_s['Realidad'] == clase]
        # Puntos CORRECTOS
        correctos = subset[subset['Acierto'] == True]
        ax.scatter(correctos['g-r'], correctos['u-g'], 
                   c=colors_ok[clase], marker=markers[clase], 
                   label=f'{clase} (Acierto)', alpha=0.3, s=30)
        # Puntos INCORRECTOS
        fallos = subset[subset['Acierto'] == False]
        ax.scatter(fallos['g-r'], fallos['u-g'], 
                   c='red', marker=markers[clase], 
                   label=f'Error de Clasificación' if clase == 'GALAXY' else "", 
                   alpha=0.9, s=50, edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel("Índice de Color (g - r)", fontsize=13)
    ax.set_ylabel("Índice de Color (u - g)", fontsize=13)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # NOTA: La anotación del 'Stellar Locus' ha sido eliminada por indicación de la tutora.

# Leyenda unificada
handles, labels = axs[0,0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
orden_deseado = ['GALAXY (Acierto)', 'STAR (Acierto)', 'QSO (Acierto)', 'Error de Clasificación']
handles_ordenados = [by_label[l] for l in orden_deseado if l in by_label]
labels_ordenados = [l for l in orden_deseado if l in by_label]

fig.legend(handles_ordenados, labels_ordenados, loc='upper center', 
           bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=14, frameon=True, markerscale=1.5)

plt.tight_layout()
# OPTIMIZACIÓN PARA OVERLEAF
plt.savefig("5_super_panel_errores.jpg", format='jpg', dpi=150, pil_kwargs={'quality': 80}, bbox_inches='tight')
plt.show()

print("\n=== PROCESO COMPLETADO: Gráficos ligeros generados (sin anotación). ===")