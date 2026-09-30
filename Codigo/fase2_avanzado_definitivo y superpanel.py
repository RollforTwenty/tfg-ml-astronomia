# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: 05_super_panel_v1_revision_ejes.py
ETAPA ORIGINAL DEL CÓDIGO: Fase 2 (Ingeniería de Características)
CORRESPONDENCIA EN LA MEMORIA: Capítulo 4, Fase 4 (Optimización Supervisada)
=============================================================================
...

=============================================================================

DESCRIPCIÓN:
Este script documenta una fase intermedia en el desarrollo de la Figura 4.1 
(Panel comparativo de errores). Constituye un refinamiento visual respecto a 
las primeras pruebas, donde se implementaron las siguientes modificaciones a 
petición de la revisión académica:
1. Re-asignación de los ejes de color termodinámicos.
2. Modificación del esquema de color base para mejorar el contraste de la 
   población de galaxias (de 'lightgray' a 'gray').

MOTIVO DE OBSOLESCENCIA:
A pesar de las mejoras visuales, el flujo de procesamiento de datos subyacente 
sigue siendo preliminar. El código emplea técnicas de submuestreo agresivo 
(recorte de clases mayoritarias) en lugar del algoritmo SMOTE detallado en la 
metodología final del TFG. Asimismo, la anotación del 'Stellar Locus' fue 
descartada en la versión definitiva debido al solapamiento físico en esa región.

NOTA: Para consultar la implementación completa con remuestreo SMOTE y 
estética final, diríjase a la carpeta de "Código Definitivo".
=============================================================================
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

# Configuración estética
sns.set_theme(style="whitegrid")

# --- 1. CARGA Y LIMPIEZA INICIAL ---
print("\n=== 1. PREPARACIÓN DE DATOS ===")
nombre_archivo = "star_classification.csv"
# Cargamos saltándonos los IDs corruptos
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object})

# Limpiamos outliers extremos para que los gráficos se vean bien
# (Hay algunos errores instrumentales con valores -9999 que ensucian el gráfico)
datos = datos[(datos['u'] > -100) & (datos['z'] > -100) & (datos['redshift'] >= 0)]

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (EL VECTOR x') ---
# Definimos los colores astronómicos (Restas de magnitudes)
datos['u-g'] = datos['u'] - datos['g']
datos['g-r'] = datos['g'] - datos['r']
datos['r-i'] = datos['r'] - datos['i']
datos['i-z'] = datos['i'] - datos['z']

print("Nuevas variables generadas (x'): u-g, g-r, r-i, i-z")

# --- 3. EQUILIBRADO DE LA MUESTRA (REMUESTREO) ---
# Queremos el mismo número de objetos de cada clase
minimo_clase = datos['class'].value_counts().min() # Debería ser ~18.000 (QSO)
print(f"Reduciendo todas las clases a {minimo_clase} muestras para equilibrar...")

# Función para equilibrar
def equilibrar_datos(df):
    df_bal = df.groupby('class').apply(lambda x: x.sample(minimo_clase, random_state=42))
    return df_bal.reset_index(drop=True)

datos_bal = equilibrar_datos(datos)
print("Conteo tras equilibrar:")
print(datos_bal['class'].value_counts())

# --- 4. VISUALIZACIÓN: DIAGRAMA COLOR-COLOR (DECISIÓN x vs x') ---
print("\n=== GENERANDO DIAGRAMAS COLOR-COLOR ===")

plt.figure(figsize=(12, 5))

# Gráfica A: Magnitudes Crudas (u vs g) - El vector x
plt.subplot(1, 2, 1)
sns.scatterplot(data=datos_bal.sample(5000), x='u', y='g', hue='class', s=10, alpha=0.6)
plt.title("Espacio Original (x): Magnitud u vs g")

# Gráfica B: Colores (u-g vs g-r) - El vector x'
plt.subplot(1, 2, 2)
sns.scatterplot(data=datos_bal.sample(5000), x='u-g', y='g-r', hue='class', s=10, alpha=0.6)
plt.title("Espacio Transformado (x'): Color u-g vs g-r")
plt.xlim(-1, 3); plt.ylim(-1, 2) # Zoom a la zona relevante

plt.tight_layout()
plt.savefig("4_comparativa_espacios.png")
print(">> Gráfico guardado: '4_comparativa_espacios.png'") 
print("OBSERVACIÓN: Fíjate cómo en la derecha (Colores) las clases se separan mejor.")


# --- 5. ENTRENAMIENTO PARA EL SUPER PANEL ---
# Usaremos las variables de COLOR (x') + Redshift
features = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']
X = datos_bal[features]
y = datos_bal['class']

# Dividimos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43, stratify=y)

# Escalado (Necesario para SVM)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# A) Modelo RF
print("\nEntrenando Random Forest (con datos equilibrados)...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_sc, y_train)

# B) Modelo SVM
print("Entrenando SVM (con datos equilibrados)...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_sc, y_train)

# --- 6. GENERACIÓN DEL "SÚPER PANEL" (4 VENTANAS) - VERSIÓN CORREGIDA ---
print("\n=== GENERANDO SÚPER PANEL DE ERRORES (VERSIÓN TUTORA) ===")

# Función auxiliar para preparar datos de ploteo
def preparar_datos_plot(modelo, X_data, y_real, nombre_set):
    y_pred = modelo.predict(X_data)
    # Des-escalamos para que el gráfico muestre valores reales de color
    df_plot = pd.DataFrame(scaler.inverse_transform(X_data), columns=features)
    
    df_plot['Realidad'] = y_real.values
    df_plot['Acierto'] = (y_real.values == y_pred) 
    df_plot['Set'] = nombre_set
    return df_plot

# Preparamos los 4 dataframes
df_rf_train = preparar_datos_plot(rf, X_train_sc, y_train, "RF - Train")
df_rf_test  = preparar_datos_plot(rf, X_test_sc,  y_test,  "RF - Test")
df_svm_train = preparar_datos_plot(svm, X_train_sc, y_train, "SVM - Train")
df_svm_test  = preparar_datos_plot(svm, X_test_sc,  y_test,  "SVM - Test")

# Configuramos el panel 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
markers = {'GALAXY': 'o', 'STAR': '^', 'QSO': 's'} 
# Colores pedidos por la tutora para los aciertos
colors_ok = {'GALAXY': 'gray', 'STAR': 'blue', 'QSO': 'green'}

lista_plots = [
    (df_rf_train, axs[0, 0], "Random Forest (Entrenamiento)"),
    (df_svm_train, axs[0, 1], "SVM (Entrenamiento)"),
    (df_rf_test, axs[1, 0], "Random Forest (Prueba)"),
    (df_svm_test, axs[1, 1], "SVM (Prueba)")
]

for df, ax, titulo in lista_plots:
    # Muestra de 2500 puntos para claridad visual
    df_s = df.sample(min(2500, len(df)), random_state=42)
    
    for clase in ['GALAXY', 'STAR', 'QSO']:
        subset = df_s[df_s['Realidad'] == clase]
        
        # 1. Puntos CORRECTOS (Colores específicos por clase)
        correctos = subset[subset['Acierto'] == True]
        ax.scatter(correctos['u-g'], correctos['g-r'], 
                   c=colors_ok[clase], marker=markers[clase], 
                   label=f'{clase} (Acierto)', alpha=0.25, s=25)
        
        # 2. Puntos INCORRECTOS (Rojo intenso para todos los fallos)
        fallos = subset[subset['Acierto'] == False]
        ax.scatter(fallos['u-g'], fallos['g-r'], 
                   c='red', marker=markers[clase], 
                   label=f'Error de Clasificación' if clase == 'GALAXY' else "", 
                   alpha=0.8, s=45, zorder=5)

    # Estética y Etiquetas
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Índice de Color u-g", fontsize=11)
    ax.set_ylabel("Índice de Color g-r", fontsize=11)
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- ANOTACIÓN STELLAR LOCUS (Petición Leire) ---
    ax.annotate('Stellar Locus', 
                xy=(1.3, 0.5), xytext=(2.0, 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# --- Leyenda unificada y formal (Ordenada) ---
handles, labels = axs[0,0].get_legend_handles_labels()

# 1. Eliminamos duplicados
by_label = dict(zip(labels, handles))

# 2. Forzamos el orden deseado en la leyenda
orden_deseado = ['GALAXY (Acierto)', 'STAR (Acierto)', 'QSO (Acierto)', 'Error de Clasificación']

# 3. Creamos listas ordenadas basadas en el orden deseado
handles_ordenados = []
labels_ordenados = []
for label in orden_deseado:
    if label in by_label:
        handles_ordenados.append(by_label[label])
        labels_ordenados.append(label)

# 4. Dibujamos la leyenda con el nuevo orden
fig.legend(handles_ordenados, labels_ordenados, loc='upper center', 
           bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=11, frameon=True)

plt.tight_layout()
# Guardamos como JPG para máxima compatibilidad con tu LaTeX
plt.savefig("5_super_panel_errores.jpg", bbox_inches='tight', dpi=300)
plt.show()