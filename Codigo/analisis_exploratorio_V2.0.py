# -*- coding: utf-8 -*-
"""
Fase 1: Análisis Exploratorio y Limpieza Estadística
TFG: Machine Learning para Astronomía
Autor: David Díez Roshan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGA DE DATOS
# ----------------------------------------------------------------
print("--- 1. CARGANDO DATOS ---")
nombre_archivo = "star_classification.csv"
# obj_ID y spec_obj_ID son texto (object) para no perder precisión
datos = pd.read_csv(nombre_archivo, dtype={'obj_ID': object, 'spec_obj_ID': object})
print(f"Datos cargados: {datos.shape[0]} filas, {datos.shape[1]} columnas.")

# 2. ANÁLISIS ESTADÍSTICO PRELIMINAR (Petición Tutora 00)
# ----------------------------------------------------------------
print("\n--- 2. ANÁLISIS DE LA VARIABLE OBJETIVO (CLASES) ---")
conteo_clases = datos['class'].value_counts()
print(conteo_clases)

# GRÁFICO 1: Barras de la distribución inicial
plt.figure(figsize=(8, 6))
sns.barplot(x=conteo_clases.index, y=conteo_clases.values, hue=conteo_clases.index, legend=False, palette="viridis")
plt.title("Distribución de Objetos Celestes (Original)", fontsize=14)
plt.xlabel("Clase")
plt.ylabel("Número de Observaciones")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("distribucion_clases_inicial.png")
print(">> Gráfico guardado: 'distribucion_clases_inicial.png'")


# 3. DETECCIÓN DE DUPLICADOS Y VARIABILIDAD (Petición Tutora 1, 3, 7)
# ----------------------------------------------------------------
print("\n--- 3. ANÁLISIS DE DUPLICADOS (BUSCANDO VARIABILIDAD) ---")

# Tu tutora sospecha que hay objetos repetidos (mismo objeto físico, varias fotos).
# Vamos a comprobarlo usando los identificadores.

# Chequeo 1: ¿Hay filas exáctamente iguales?
duplicados_totales = datos.duplicated().sum()
print(f"Filas totalmente duplicadas: {duplicados_totales}")

# Chequeo 2: ¿Hay objetos con el mismo ID (obj_ID)?
# obj_ID suele ser único por "foto", pero veamos.
objetos_unicos = datos['obj_ID'].nunique()
print(f"Total filas: {len(datos)}")
print(f"Objetos únicos (obj_ID): {objetos_unicos}")

if objetos_unicos < len(datos):
    print("¡ALERTA! Hay obj_ID repetidos. Podemos estudiar variabilidad.")
else:
    print("RESULTADO: Cada fila tiene un obj_ID único.")
    print("NOTA PARA LA TUTORA: En este dataset de Kaggle, parece que ya han")
    print("filtrado las repeticiones o cada fila es una detección única.")

# Chequeo 3: Buscamos por posición (misma Alpha y Delta)
# A veces el ID cambia pero la posición es la misma (dentro de un margen de error).
# Esto es más avanzado, lo dejamos anotado aquí.


# 4. LIMPIEZA DE COLUMNAS (Petición Tutora 8 y 5)
# ----------------------------------------------------------------
print("\n--- 4. SELECCIÓN DE VARIABLES (LIMPIEZA) ---")

# Lista de columnas a mantener (Física pura)
# Tu tutora pidió explícitamente usar u, g, r, i, z y redshift.
col_fisicas = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']

# Creamos el dataset limpio solo con esto
datos_limpios = datos[col_fisicas].copy()

print("Eliminando coordenadas (alpha, delta) y metadatos (IDs)...")
print("Variables finales para el modelo:")
print(datos_limpios.columns.tolist())

print("\n--- Primeras 5 filas del dataset limpio ---")
print(datos_limpios.head())

# 5. GUARDAR DATOS LIMPIOS (Opcional, para usar luego)
# ----------------------------------------------------------------
# datos_limpios.to_csv("datos_procesados_tfg.csv", index=False)


# 6. ANÁLISIS DE VARIABILIDAD (Puntos 3, 4, 7 de la Tutora)
# ----------------------------------------------------------------
print("\n--- 6. ESTUDIO DE VARIABILIDAD DE OBJETOS REPETIDOS ---")

# 1. Identificar objetos con múltiples observaciones
conteo_por_obj = datos.groupby('obj_ID').size()
ids_repetidos = conteo_por_obj[conteo_por_obj > 1].index

print(f"Objetos con más de 1 observación: {len(ids_repetidos)}")

# 2. Calcular la desviación estándar (std) de las magnitudes para esos objetos
#    Si std es alta, el objeto brilla distinto en cada foto -> VARIABLE
#    Si std es baja (cercana a 0.02), es ruido -> NO VARIABLE
columnas_mag = ['u', 'g', 'r', 'i', 'z']

# Filtramos solo los datos de los objetos repetidos
datos_repetidos = datos[datos['obj_ID'].isin(ids_repetidos)]

# Agrupamos por ID y calculamos la desviación estándar
variabilidad = datos_repetidos.groupby('obj_ID')[columnas_mag].std()

# 3. Aplicar criterio de variabilidad
#    Tu tutora mencionó "varianza < 0.4" (o desviación < sqrt(0.4) = 0.63)
#    También mencionó el error fotométrico (0.01 - 0.02).
#    Vamos a ser conservadores: Si la desviación es mayor que 0.05, algo pasa.

# Calculamos la varianza máxima en cualquiera de las bandas para cada objeto
variabilidad['max_std'] = variabilidad.max(axis=1)
variabilidad['max_var'] = variabilidad['max_std'] ** 2 # La varianza es el cuadrado de la std

print("\nEjemplo de variabilidad calculada (primeras 5 filas):")
print(variabilidad.head())

# 4. Clasificar: Variable vs Estable
# UMBRAL: Usamos 0.05 como corte inicial (aprox 2-3 veces el error del telescopio)
umbral_corte = 0.05 
ids_variables = variabilidad[variabilidad['max_std'] > umbral_corte].index
ids_estables = variabilidad[variabilidad['max_std'] <= umbral_corte].index

print(f"\n--- RESULTADOS DEL ANÁLISIS ---")
print(f"Umbral de corte (std): {umbral_corte} magnitudes")
print(f"Objetos REPETIDOS considerados VARIABLES: {len(ids_variables)}")
print(f"Objetos REPETIDOS considerados ESTABLES:  {len(ids_estables)}")

# 5. Visualizar la varianza (Histograma)
plt.figure(figsize=(10, 6))
sns.histplot(variabilidad['max_var'], bins=50, kde=True, log_scale=(False, True)) # Escala log en Y para ver mejor
plt.axvline(0.4, color='red', linestyle='--', label='Umbral Tutora (Var=0.4)')
plt.title("Distribución de la Varianza en Objetos Repetidos")
plt.xlabel("Máxima Varianza en bandas (u,g,r,i,z)")
plt.ylabel("Frecuencia (Log)")
plt.legend()
plt.savefig("distribucion_varianza.png")
print(">> Gráfico guardado: 'distribucion_varianza.png'")

# NOTA PARA EL TFG:
# Aquí decidiremos qué hacer:
# - A los ESTABLES: Los "unificamos" (hacemos la media de sus valores).
# - A los VARIABLES: O los quitamos, o creamos una etiqueta nueva.