"""
DESCARGADOR DE DATASET DESDE KAGGLE
Complementa las imágenes de FiftyOne con datasets de Kaggle
Especialmente útil para tortugas, hormigas y mariquitas
"""

import os
import sys
import zipfile
import shutil
from PIL import Image
from pathlib import Path

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "dataset_balanceado")
temp_dir = os.path.join(base_dir, "temp_kaggle")
TARGET_POR_CLASE = 15000  # Reducido a 15,000

# Mapeo de datasets de Kaggle por clase - MAXIMIZADO PARA MÁS IMÁGENES REALES
KAGGLE_DATASETS = {
    "gatos": [
        "crawford/cat-dataset",  # Cats dataset (~10k imágenes)
        "shaunthesheep/microsoft-catsvsdogs-dataset",  # Microsoft Cats (~12k)
    ],
    "perros": [
        "jessicali9530/stanford-dogs-dataset",  # Stanford Dogs (~20k)
        "shaunthesheep/microsoft-catsvsdogs-dataset",  # Microsoft Dogs (~12k)
    ],
    "tortugas": [
        "iamsouravbanerjee/animal-image-dataset-90-different-animals",  # 90 animals (incluye tortugas)
        "antoreepjana/animals-detection-images-dataset",  # Animals detection
        "gpiosenka/animals-detection-images-dataset",  # Animals with boxes
        "alessiocorrado99/animals10",  # 10 animals dataset
    ],
    "hormigas": [
        "thedatasith/hymenoptera",  # ⭐ Hormigas/abejas (~2500 hormigas) - PÚBLICO
        "jerzydziewierz/ants-bees-dataset",  # ⭐ Ants vs Bees - PÚBLICO
        "vencerlanz09/insects-classification-dataset",  # ⭐ Insects classification
        "mistag/insect-images-for-detecting-and-visualizing",  # ⭐ Insect detection
        "veeralakrishna/butterfly-dataset",  # Insects general (puede incluir hormigas)
    ],
    "mariquitas": [
        "vencerlanz09/insects-classification-dataset",  # ⭐ Insects classification
        "mistag/insect-images-for-detecting-and-visualizing",  # ⭐ Insect detection
        "veeralakrishna/butterfly-dataset",  # Insects dataset
        "jerzydziewierz/ants-bees-dataset",  # Insects dataset
    ]
}

# Nombres alternativos para buscar en carpetas de datasets descargados - EXPANDIDO
NOMBRES_ALTERNATIVOS = {
    "gatos": ["cat", "cats", "Cat", "Cats", "feline", "kitten", "kitty", "gato", "gatos"],
    "perros": ["dog", "dogs", "Dog", "Dogs", "canine", "puppy", "perro", "perros"],
    "tortugas": ["turtle", "tortoise", "Turtle", "Tortoise", "sea_turtle", "tortuga", "tortugas", "terrapin"],
    "hormigas": ["ant", "ants", "Ant", "Ants", "ANT", "ANTS", "hymenoptera", "Hymenoptera", "formicidae", "Formicidae", "hormiga", "hormigas", "fire_ant", "fireant"],
    "mariquitas": ["ladybug", "ladybird", "Ladybug", "Ladybird", "LADYBUG", "ladybeetle", "lady_beetle", "coccinellidae", "Coccinellidae", "mariquita", "beetle", "Beetle"]
}

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def verificar_kaggle_api():
    """Verifica si Kaggle API está instalada y configurada"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except ImportError:
        print("❌ ERROR: Kaggle API no está instalada")
        print("   Instala con: pip install kaggle")
        return None
    except Exception as e:
        print("❌ ERROR: Kaggle API no está configurada correctamente")
        print(f"   {str(e)}")
        print("\n📝 INSTRUCCIONES:")
        print("   1. Crea cuenta en https://www.kaggle.com")
        print("   2. Ve a Account → Create New API Token")
        print("   3. Guarda kaggle.json en:")
        print("      Windows: C:\\Users\\<usuario>\\.kaggle\\kaggle.json")
        print("      Linux/Mac: ~/.kaggle/kaggle.json")
        return None

def buscar_imagenes_en_carpeta(carpeta, clase):
    """Busca imágenes de una clase en una carpeta descargada - BÚSQUEDA AGRESIVA"""
    imagenes_encontradas = []

    if not os.path.exists(carpeta):
        return imagenes_encontradas

    # Buscar por nombres alternativos
    nombres_buscar = NOMBRES_ALTERNATIVOS.get(clase, [clase])

    for root, dirs, files in os.walk(carpeta):
        # Verificar si el nombre de la carpeta coincide con algún nombre alternativo
        carpeta_nombre = os.path.basename(root).lower()
        carpeta_path_completo = root.lower()

        coincide = False
        for nombre_alt in nombres_buscar:
            nombre_lower = nombre_alt.lower()
            # Buscar en el nombre de la carpeta actual O en la ruta completa
            if nombre_lower in carpeta_nombre or nombre_lower in carpeta_path_completo:
                coincide = True
                break

        if coincide:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    ruta_completa = os.path.join(root, file)
                    imagenes_encontradas.append(ruta_completa)
        else:
            # BÚSQUEDA ADICIONAL: Verificar nombres de archivos también
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_lower = file.lower()
                    for nombre_alt in nombres_buscar:
                        if nombre_alt.lower() in file_lower:
                            ruta_completa = os.path.join(root, file)
                            imagenes_encontradas.append(ruta_completa)
                            break

    return imagenes_encontradas

def procesar_imagen(img_path, clase, contador, output_dir):
    """Procesa y guarda una imagen en el formato correcto"""
    try:
        with Image.open(img_path) as img:
            # Convertir a RGB
            img = img.convert('RGB')

            # Redimensionar a 64×64 (mejorado para mejor accuracy)
            img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)

            # Guardar
            nombre_archivo = f"{clase}_kaggle_{contador:06d}.jpg"
            ruta_destino = os.path.join(output_dir, clase, nombre_archivo)

            img_resized.save(ruta_destino, quality=95)
            return True
    except Exception as e:
        return False

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def descargar_desde_kaggle():
    print("="*80)
    print("📦 DESCARGADOR DE KAGGLE - COMPLEMENTO AL DATASET")
    print("="*80)
    print(f"📂 Destino: {output_dir}")
    print(f"🎯 Meta: {TARGET_POR_CLASE:,} imágenes por clase")
    print()

    # Verificar API de Kaggle
    api = verificar_kaggle_api()
    if not api:
        print("\n⚠️  No se puede continuar sin Kaggle API configurada")
        print("   Primero ejecuta: python generar_dataset_imagenes_completas.py")
        return

    print("✅ Kaggle API configurada correctamente")
    print()

    # Crear carpeta temporal
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    # Procesar cada clase
    for clase, datasets in KAGGLE_DATASETS.items():
        print(f"\n{'='*80}")
        print(f"🔍 Procesando: {clase.upper()}")
        print(f"{'='*80}")

        # Verificar cuántas imágenes ya tenemos
        clase_dir = os.path.join(output_dir, clase)
        if not os.path.exists(clase_dir):
            os.makedirs(clase_dir, exist_ok=True)

        imagenes_existentes = len([f for f in os.listdir(clase_dir)
                                  if f.endswith(('.jpg', '.png'))])

        print(f"   Imágenes actuales: {imagenes_existentes:,}")

        if imagenes_existentes >= TARGET_POR_CLASE:
            print(f"   ✅ YA COMPLETO - Omitiendo")
            continue

        faltantes = TARGET_POR_CLASE - imagenes_existentes
        print(f"   ⚠️  Faltan: {faltantes:,} imágenes")

        # Intentar descargar de cada dataset de Kaggle
        agregadas_total = 0

        for dataset_name in datasets:
            if agregadas_total >= faltantes:
                break

            print(f"\n   📥 Descargando dataset: {dataset_name}")

            try:
                # Crear carpeta para este dataset
                dataset_temp = os.path.join(temp_dir, dataset_name.replace("/", "_"))

                # Descargar dataset
                api.dataset_download_files(
                    dataset_name,
                    path=dataset_temp,
                    unzip=True,
                    quiet=False
                )

                print(f"   ✅ Descargado correctamente")

                # Buscar imágenes de esta clase
                imagenes_encontradas = buscar_imagenes_en_carpeta(dataset_temp, clase)
                print(f"   🔍 Imágenes encontradas: {len(imagenes_encontradas):,}")

                if len(imagenes_encontradas) == 0:
                    print(f"   ⚠️  No se encontraron imágenes de {clase}")
                    continue

                # Procesar imágenes
                print(f"   🔧 Procesando imágenes...")
                procesadas = 0
                errores = 0

                for img_path in imagenes_encontradas:
                    if agregadas_total >= faltantes:
                        break

                    contador = imagenes_existentes + agregadas_total

                    if procesar_imagen(img_path, clase, contador, output_dir):
                        procesadas += 1
                        agregadas_total += 1

                        if procesadas % 500 == 0:
                            print(f"      ✅ {procesadas:,} procesadas")
                    else:
                        errores += 1

                print(f"   ✅ Agregadas: {procesadas:,} imágenes")
                if errores > 0:
                    print(f"   ⚠️  Errores: {errores:,}")

                # Limpiar carpeta temporal de este dataset
                try:
                    shutil.rmtree(dataset_temp)
                except:
                    pass

            except Exception as e:
                print(f"   ❌ ERROR al descargar {dataset_name}: {str(e)[:100]}")
                continue

        # Reporte final de esta clase
        total_final = imagenes_existentes + agregadas_total
        print(f"\n   📊 RESUMEN {clase.upper()}:")
        print(f"      Antes: {imagenes_existentes:,}")
        print(f"      Agregadas desde Kaggle: {agregadas_total:,}")
        print(f"      Total ahora: {total_final:,} / {TARGET_POR_CLASE:,}")

    # Limpiar carpeta temporal completa
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except:
        pass

    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL DESPUÉS DE KAGGLE")
    print("="*80)

    for clase in KAGGLE_DATASETS.keys():
        clase_dir = os.path.join(output_dir, clase)
        if os.path.exists(clase_dir):
            total = len([f for f in os.listdir(clase_dir)
                        if f.endswith(('.jpg', '.png'))])
            status = "✅" if total >= TARGET_POR_CLASE else "⚠️"
            porcentaje = (total / TARGET_POR_CLASE) * 100
            print(f"   {status} {clase:12} : {total:,} / {TARGET_POR_CLASE:,} ({porcentaje:.1f}%)")

    print("\n💡 PRÓXIMO PASO:")
    print("   Si aún faltan imágenes, ejecuta:")
    print("   python completar_dataset_con_augmentation.py")

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    descargar_desde_kaggle()
