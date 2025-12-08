"""
Script para COMPLETAR el dataset usando Data Augmentation
Genera imágenes sintéticas para las clases con pocas imágenes
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset_balanceado")
TARGET_POR_CLASE = 15000  # Reducido a 15,000

# Data Augmentation OPTIMIZADO - Transformaciones más naturales
# Parámetros ajustados para generar imágenes realistas sin distorsión excesiva
datagen = ImageDataGenerator(
    rotation_range=25,              # Reducido de 40° a 25° (más natural)
    width_shift_range=0.15,         # Reducido de 0.3 a 0.15
    height_shift_range=0.15,        # Reducido de 0.3 a 0.15
    shear_range=0.1,                # Reducido de 0.3 a 0.1 (menos distorsión)
    zoom_range=0.2,                 # Reducido de 0.3 a 0.2
    horizontal_flip=True,           # Mantener (natural para animales)
    vertical_flip=False,            # DESACTIVADO (animales no están boca abajo)
    brightness_range=[0.8, 1.2],    # Reducido rango de 0.7-1.3 a 0.8-1.2
    fill_mode='reflect'             # Cambiado de 'nearest' a 'reflect' (más natural)
)

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def completar_dataset():
    print("="*80)
    print("🔧 COMPLETADOR DE DATASET CON DATA AUGMENTATION")
    print("="*80)
    print(f"📂 Dataset: {dataset_dir}")
    print(f"🎯 Meta: {TARGET_POR_CLASE:,} imágenes por clase")
    print()

    if not os.path.exists(dataset_dir):
        print("❌ ERROR: No existe la carpeta dataset_balanceado")
        print("   Primero ejecuta: python generar_dataset_imagenes_completas.py")
        return

    # Obtener todas las clases
    clases = [d for d in os.listdir(dataset_dir)
              if os.path.isdir(os.path.join(dataset_dir, d))]

    print(f"📊 Clases encontradas: {clases}")
    print()

    for clase in clases:
        clase_dir = os.path.join(dataset_dir, clase)

        # Contar imágenes existentes
        imagenes_existentes = [f for f in os.listdir(clase_dir)
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
        num_existentes = len(imagenes_existentes)

        print(f"\n{'='*80}")
        print(f"🔍 Procesando: {clase.upper()}")
        print(f"{'='*80}")
        print(f"   Imágenes actuales: {num_existentes:,}")
        print(f"   Meta: {TARGET_POR_CLASE:,}")

        if num_existentes >= TARGET_POR_CLASE:
            print(f"   ✅ YA COMPLETO - No se necesita augmentation")
            continue

        faltantes = TARGET_POR_CLASE - num_existentes
        print(f"   ⚠️  Faltan: {faltantes:,} imágenes")
        print(f"   🔧 Generando imágenes sintéticas...")

        # Cargar todas las imágenes reales
        print(f"   📂 Cargando {num_existentes:,} imágenes originales...")
        imagenes_originales = []

        for img_file in imagenes_existentes:
            try:
                img_path = os.path.join(clase_dir, img_file)
                img = Image.open(img_path)
                img_array = np.array(img)

                # Asegurar que sea RGB 64×64 (mejorado para mejor accuracy)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                imagenes_originales.append(img_array)
            except Exception as e:
                continue

        if len(imagenes_originales) == 0:
            print(f"   ❌ ERROR: No se pudieron cargar imágenes de {clase}")
            continue

        print(f"   ✅ Cargadas {len(imagenes_originales):,} imágenes")

        # Convertir a numpy array
        imagenes_originales = np.array(imagenes_originales)

        # Generar imágenes sintéticas
        print(f"   🎨 Generando {faltantes:,} imágenes sintéticas...")

        generadas = 0
        intentos = 0
        max_intentos = faltantes * 3  # Margen de seguridad

        while generadas < faltantes and intentos < max_intentos:
            # Seleccionar imagen aleatoria
            idx = random.randint(0, len(imagenes_originales) - 1)
            img_original = imagenes_originales[idx]

            # Reshape para el generador (añadir dimensión de batch)
            img_batch = img_original.reshape((1,) + img_original.shape)

            # Generar imagen transformada
            try:
                # Aplicar transformación aleatoria
                for batch in datagen.flow(img_batch, batch_size=1):
                    img_augmented = batch[0].astype(np.uint8)

                    # Guardar imagen sintética
                    nombre_archivo = f"{clase}_synthetic_{num_existentes + generadas:06d}.jpg"
                    ruta_guardado = os.path.join(clase_dir, nombre_archivo)

                    img_pil = Image.fromarray(img_augmented)
                    img_pil.save(ruta_guardado, quality=95)

                    generadas += 1

                    # Progreso cada 500 imágenes
                    if generadas % 500 == 0:
                        porcentaje = (generadas / faltantes) * 100
                        print(f"      [{generadas:,}/{faltantes:,}] {porcentaje:.1f}%")

                    break  # Solo queremos 1 imagen de este batch

            except Exception as e:
                pass

            intentos += 1

        total_final = num_existentes + generadas
        print(f"\n   ✅ {clase.upper()} COMPLETADO")
        print(f"      Originales: {num_existentes:,}")
        print(f"      Sintéticas: {generadas:,}")
        print(f"      Total: {total_final:,} / {TARGET_POR_CLASE:,}")

        if total_final < TARGET_POR_CLASE:
            print(f"      ⚠️  Aún faltan {TARGET_POR_CLASE - total_final:,}")

    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL DEL DATASET COMPLETADO")
    print("="*80)

    for clase in clases:
        clase_dir = os.path.join(dataset_dir, clase)
        num_imgs = len([f for f in os.listdir(clase_dir)
                       if f.endswith(('.jpg', '.png'))])

        # Contar originales y sintéticas
        originales = len([f for f in os.listdir(clase_dir)
                         if not 'synthetic' in f and f.endswith(('.jpg', '.png'))])
        sinteticas = num_imgs - originales

        status = "✅" if num_imgs >= TARGET_POR_CLASE else "⚠️"
        print(f"{status} {clase:12} : {num_imgs:,} / {TARGET_POR_CLASE:,} "
              f"(Real: {originales:,}, Sintética: {sinteticas:,})")

    print("="*80)

    # Verificar si está completo
    todas_completas = True
    for clase in clases:
        clase_dir = os.path.join(dataset_dir, clase)
        num_imgs = len([f for f in os.listdir(clase_dir)
                       if f.endswith(('.jpg', '.png'))])
        if num_imgs < TARGET_POR_CLASE:
            todas_completas = False
            break

    if todas_completas:
        print("\n🎉🎉🎉 DATASET PERFECTAMENTE BALANCEADO!")
        print(f"✅ Total: {TARGET_POR_CLASE * len(clases):,} imágenes")
        print("\n💡 PRÓXIMO PASO:")
        print("   1. Abre CNN.ipynb")
        print("   2. Kernel → Restart & Run All")
        print("   3. Espera el entrenamiento (~45-60 min)")
    else:
        print("\n⚠️  Dataset aún incompleto")
        print("   Ejecuta este script nuevamente para completar")

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    completar_dataset()
