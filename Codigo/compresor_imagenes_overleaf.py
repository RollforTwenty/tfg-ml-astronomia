# -*- coding: utf-8 -*-
"""
=============================================================================
ARCHIVO: util_compresor_imagenes_overleaf.py
TIPO: Script de Utilería / Automatización
=============================================================================

DESCRIPCIÓN:
Este script automatiza la compresión y conversión por lotes de las gráficas 
generadas durante el análisis exploratorio y el entrenamiento de modelos. 
Las librerías de visualización (Matplotlib/Seaborn) exportan imágenes PNG 
de alta resolución con canales alfa (transparencia), lo que genera archivos 
de gran tamaño que exceden los límites de compilación y almacenamiento 
de plataformas como Overleaf.

PROCEDIMIENTO:
1. Escanea el directorio activo en busca de archivos .png.
2. Elimina el canal alfa sustituyéndolo por un fondo sólido (blanco).
3. Aplica un redimensionamiento condicional (downscaling) con filtro LANCZOS 
   a aquellas imágenes cuyo ancho supere los 1200 píxeles.
4. Exporta las imágenes resultantes en formato JPG optimizado (calidad 85).

RESULTADO:
Se genera un directorio 'imagenes_ligeras_overleaf' con los recursos gráficos 
listos para ser integrados en el documento LaTeX principal sin comprometer 
los tiempos de renderizado.
=============================================================================
"""

import os
from PIL import Image

# Configuración
directorio_origen = "."  # La carpeta donde estás
directorio_destino = "imagenes_ligeras_overleaf"

# Crear carpeta de destino si no existe
if not os.path.exists(directorio_destino):
    os.makedirs(directorio_destino)

print(f"--- Comprimiendo imágenes en: {os.getcwd()} ---")

# Buscar todas las imágenes PNG
archivos = [f for f in os.listdir(directorio_origen) if f.endswith('.png')]

if not archivos:
    print("⚠️ No he encontrado archivos .png en esta carpeta.")
else:
    for archivo in archivos:
        try:
            # Abrir imagen
            ruta_completa = os.path.join(directorio_origen, archivo)
            img = Image.open(ruta_completa)
            
            # Si tiene fondo transparente, ponerlo blanco (necesario para JPG)
            if img.mode in ('RGBA', 'LA'):
                fondo = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                fondo.paste(img, img.split()[-1])
                img = fondo.convert('RGB')
            else:
                img = img.convert('RGB')
                
            # Reducir tamaño si es gigante (más de 1200px es innecesario para un TFG)
            if img.width > 1200:
                ratio = 1200 / float(img.width)
                altura = int((float(img.height) * float(ratio)))
                img = img.resize((1200, altura), Image.Resampling.LANCZOS)
            
            # Guardar como JPG (Calidad 85 es perfecta para documentos)
            nombre_nuevo = os.path.splitext(archivo)[0] + ".jpg"
            ruta_guardado = os.path.join(directorio_destino, nombre_nuevo)
            
            img.save(ruta_guardado, "JPEG", quality=85, optimize=True)
            
            print(f"✅ {archivo} -> Guardado como JPG optimizado")
            
        except Exception as e:
            print(f"❌ Error con {archivo}: {e}")

    print(f"\n¡LISTO! Tus imágenes ligeras están en la carpeta: {directorio_destino}")