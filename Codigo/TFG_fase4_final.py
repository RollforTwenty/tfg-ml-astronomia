# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 05_optimizacion_smote_y_jerarquico.py
FASE DEL TFG: Fase 4 - Optimización Final y Clasificación Jerárquica
Estrategias Avanzadas (SMOTE Correcto + Jerarquía)
Objetivo: Superar el 97.8% de precisión de forma científica y rigurosa.
=============================================================================

DESCRIPCIÓN:
Este es el script principal de producción de resultados del TFG. Implementa 
la estrategia de optimización definitiva para alcanzar la máxima capacidad 
de generalización del modelo Random Forest sobre el catálogo SDSS.

PROCEDIMIENTO:
1. División Estratificada: Se garantiza que el conjunto de prueba (Test) 
   sea una representación pura de la realidad física, sin datos sintéticos.
2. Estrategia SMOTE: Aplicación de sobremuestreo sintético exclusivamente 
   en el conjunto de entrenamiento para equilibrar la densidad de los 
   cuásares (QSO) y estrellas (STAR).
3. Clasificación en Cascada: Implementación de un modelo jerárquico de dos 
   niveles (Nivel 1: Galaxias vs Resto; Nivel 2: Estrellas vs Cuásares).

RESULTADOS:
El script genera la Figura 4.5 de la memoria, comparando las matrices de 
confusión antes y después de la optimización, demostrando la recuperación 
drástica de la clase QSO y la estabilización del Recall en la clase STAR.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE # pip install imbalanced-learn

# 1. CARGA Y PREPARACIÓN
print("--- 1. Carga y Feature Engineering ---")
nombre_archivo = "star_classification.csv"
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})
datos = datos[(datos['u'] > -50) & (datos['z'] > -50) & (datos['redshift'] >= 0)]

# Ingeniería de características (Tus variables ganadoras x')
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos[features]
y = datos['class']

# --- AQUÍ ESTÁ LA CLAVE DEL ÉXITO ---
# Dividimos ANTES de hacer nada. El Test se queda puro (Realidad).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Datos originales Train: {len(X_train)}")
print(f"Datos originales Test:  {len(X_test)}")

# =============================================================================
# ESTRATEGIA 1: SMOTE (Data Augmentation CIENTÍFICO)
# =============================================================================
print("\n=== ESTRATEGIA 1: Random Forest con SMOTE (Solo en Train) ===")
# Solo inventamos datos en el entrenamiento para que el modelo estudie mejor
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Distribución tras SMOTE (Solo Train): {y_train_smote.value_counts().to_dict()}")

# Entrenamos
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)

# Evaluamos con el Test PURO (sin datos falsos)
y_pred_smote = rf_smote.predict(X_test)
acc_smote = accuracy_score(y_test, y_pred_smote)

print(f"--> Precisión Global (SMOTE): {acc_smote:.4f}")
print("Reporte por Clases (Fíjate en QSO):")
print(classification_report(y_test, y_pred_smote))

# =============================================================================
# ESTRATEGIA 2: MODELO JERÁRQUICO (Divide y Vencerás)
# =============================================================================
print("\n=== ESTRATEGIA 2: Clasificación en Cascada ===")

# --- NIVEL 1: ¿Es Galaxia o No? ---
# Creamos etiquetas binarias temporales
y_train_L1 = y_train.apply(lambda x: 'GALAXY' if x == 'GALAXY' else 'OTHER')
y_test_L1  = y_test.apply(lambda x: 'GALAXY' if x == 'GALAXY' else 'OTHER')

print("Entrenando Nivel 1 (Galaxia vs Resto)...")
rf_L1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_L1.fit(X_train, y_train_L1)

# --- NIVEL 2: ¿Estrella o Quásar? ---
# Filtramos para entrenar solo con lo que NO es Galaxia
mask_no_gal_train = y_train != 'GALAXY'
X_train_L2 = X_train[mask_no_gal_train]
y_train_L2 = y_train[mask_no_gal_train]

print("Entrenando Nivel 2 (Star vs QSO) con SMOTE...")
# Aplicamos SMOTE también aquí porque hay pocas estrellas comparadas con Quásares
X_train_L2_sm, y_train_L2_sm = smote.fit_resample(X_train_L2, y_train_L2)

rf_L2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_L2.fit(X_train_L2_sm, y_train_L2_sm)

# --- PREDICCIÓN CONJUNTA ---
print("Ejecutando Cascada...")
# 1. Pasamos todo por el filtro de Galaxias
pred_L1 = rf_L1.predict(X_test)

# 2. Donde dijo "OTHER", pasamos el segundo modelo
final_preds = pred_L1.copy()
mask_others_test = (pred_L1 == 'OTHER')

if mask_others_test.sum() > 0:
    pred_L2 = rf_L2.predict(X_test[mask_others_test])
    final_preds[mask_others_test] = pred_L2

acc_hier = accuracy_score(y_test, final_preds)
print(f"--> Precisión Global (Jerárquico): {acc_hier:.4f}")

# =============================================================================
# COMPARATIVA VISUAL (CORREGIDA PARA LA TUTORA: SIN SMOTE vs CON SMOTE)
# =============================================================================
print("\n=== GENERANDO MATRICES DE CONFUSIÓN COMPARATIVAS ===")

# 1. ENTRENAMOS UN MODELO BASE (SIN SMOTE) RÁPIDO PARA PODER COMPARAR EL "ANTES"
print("Entrenando modelo base (sin SMOTE) para la comparativa...")
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

# 2. CONFIGURAMOS EL GRÁFICO 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
clases = ['GALAXY', 'QSO', 'STAR'] # Orden lógico

# --- Matriz ANTES de SMOTE ---
cm_base = confusion_matrix(y_test, y_pred_base, labels=clases)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], 
            xticklabels=clases, yticklabels=clases, annot_kws={"size": 13})
axes[0].set_title('RF Original (SIN SMOTE)', fontsize=15, fontweight='bold', pad=15)
axes[0].set_ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')

# --- Matriz DESPUÉS de SMOTE (rf_smote ya estaba entrenado arriba en tu código) ---
cm_smote = confusion_matrix(y_test, y_pred_smote, labels=clases)
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1], 
            xticklabels=clases, yticklabels=clases, annot_kws={"size": 13})
axes[1].set_title('RF Optimizado (CON SMOTE)', fontsize=15, fontweight='bold', pad=15)
axes[1].set_ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("15_resultado_final_fase4.jpg", bbox_inches='tight', dpi=300)
plt.show()

print(">> Imagen '15_resultado_final_fase4.jpg' generada con éxito y lista para Overleaf.")