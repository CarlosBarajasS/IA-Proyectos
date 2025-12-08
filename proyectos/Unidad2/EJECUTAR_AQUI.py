"""
SCRIPT MAESTRO - GENERADOR COMPLETO DE DATASET
Orquesta todo el proceso de descarga y augmentation
Ejecuta automáticamente los 3 pasos en el orden correcto
"""

import os
import sys
import subprocess
import time
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
TARGET_POR_CLASE = 15000  # Reducido a 15,000

# Rutas de los scripts
SCRIPT_FIFTYONE = os.path.join(base_dir, "_paso1_fiftyone.py")
SCRIPT_KAGGLE = os.path.join(base_dir, "_paso2_kaggle.py")
SCRIPT_AUGMENTATION = os.path.join(base_dir, "_paso3_augmentation.py")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def imprimir_encabezado(titulo):
    """Imprime un encabezado decorado"""
    print("\n")
    print("="*80)
    print(f"  {titulo}")
    print("="*80)
    print()

def ejecutar_script(script_path, nombre):
    """Ejecuta un script de Python y muestra el resultado"""
    print(f"▶️  Ejecutando: {nombre}")
    print(f"   Script: {os.path.basename(script_path)}")
    print()

    inicio = time.time()

    try:
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            cwd=base_dir
        )

        duracion = time.time() - inicio

        if result.returncode == 0:
            print(f"\n✅ {nombre} completado exitosamente")
            print(f"   Duración: {duracion:.1f} segundos ({duracion/60:.1f} minutos)")
            return True
        else:
            print(f"\n⚠️  {nombre} finalizó con código de salida {result.returncode}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR al ejecutar {nombre}: {str(e)}")
        return False

def obtener_estadisticas_dataset():
    """Obtiene estadísticas del dataset actual"""
    dataset_dir = os.path.join(base_dir, "dataset_balanceado")

    if not os.path.exists(dataset_dir):
        return {}

    clases = ["gatos", "perros", "tortugas", "hormigas", "mariquitas"]
    estadisticas = {}

    for clase in clases:
        clase_dir = os.path.join(dataset_dir, clase)
        if os.path.exists(clase_dir):
            num_imgs = len([f for f in os.listdir(clase_dir)
                          if f.endswith(('.jpg', '.png'))])
            estadisticas[clase] = num_imgs
        else:
            estadisticas[clase] = 0

    return estadisticas

def mostrar_estadisticas(titulo, stats):
    """Muestra estadísticas del dataset"""
    print(f"\n{titulo}")
    print("-" * 60)

    total = 0
    for clase, count in stats.items():
        porcentaje = (count / TARGET_POR_CLASE) * 100 if TARGET_POR_CLASE > 0 else 0
        status = "✅" if count >= TARGET_POR_CLASE else "⚠️"
        print(f"{status} {clase:12} : {count:7,} / {TARGET_POR_CLASE:,} ({porcentaje:5.1f}%)")
        total += count

    print("-" * 60)
    print(f"   TOTAL        : {total:7,} / {TARGET_POR_CLASE * 5:,}")
    print()

def verificar_prerequisitos():
    """Verifica que todo esté listo para ejecutar"""
    print("🔍 Verificando prerequisitos...")

    errores = []

    # Verificar que existan los scripts
    scripts_necesarios = [
        (SCRIPT_FIFTYONE, "generar_dataset_imagenes_completas.py"),
        (SCRIPT_KAGGLE, "descargar_dataset_kaggle.py"),
        (SCRIPT_AUGMENTATION, "completar_dataset_con_augmentation.py")
    ]

    for script_path, nombre in scripts_necesarios:
        if not os.path.exists(script_path):
            errores.append(f"No se encuentra: {nombre}")

    # Verificar librerías
    try:
        import fiftyone
        print("   ✅ FiftyOne instalado")
    except ImportError:
        errores.append("FiftyOne no está instalado (pip install fiftyone)")

    try:
        import PIL
        print("   ✅ Pillow instalado")
    except ImportError:
        errores.append("Pillow no está instalado (pip install pillow)")

    try:
        import tensorflow
        print("   ✅ TensorFlow instalado")
    except ImportError:
        errores.append("TensorFlow no está instalado (pip install tensorflow)")

    # Kaggle es opcional
    try:
        import kaggle
        print("   ✅ Kaggle API instalado")
        kaggle_disponible = True
    except ImportError:
        print("   ⚠️  Kaggle API no instalado (opcional)")
        kaggle_disponible = False

    if errores:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errores:
            print(f"   - {error}")
        return False

    print("\n✅ Todos los prerequisitos están listos")
    return kaggle_disponible

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def generar_dataset_completo():
    """Función principal que orquesta todo el proceso"""

    imprimir_encabezado("🚀 GENERADOR COMPLETO DE DATASET - SISTEMA MULTI-FUENTE")

    print("📋 Este script ejecutará automáticamente:")
    print("   1. Descarga de FiftyOne (Open Images V7)")
    print("   2. Descarga de Kaggle (complemento)")
    print("   3. Data Augmentation (completar gaps)")
    print()
    print(f"🎯 Meta: {TARGET_POR_CLASE:,} imágenes por cada una de las 5 clases")
    print(f"   Total objetivo: {TARGET_POR_CLASE * 5:,} imágenes")
    print()

    # Verificar prerequisitos
    kaggle_disponible = verificar_prerequisitos()

    # Mostrar estado inicial
    stats_inicial = obtener_estadisticas_dataset()
    if stats_inicial:
        mostrar_estadisticas("📊 ESTADO INICIAL DEL DATASET", stats_inicial)
    else:
        print("📊 No existe dataset previo - Comenzando desde cero")
        print()

    # Pedir confirmación
    print("⏳ IMPORTANTE: Este proceso puede tomar varias horas")
    print("   - Descarga de FiftyOne: ~30-60 min por clase")
    print("   - Descarga de Kaggle: ~10-20 min por dataset")
    print("   - Data Augmentation: ~5-15 min por clase")
    print()

    respuesta = input("¿Deseas continuar? (s/n): ").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Proceso cancelado por el usuario")
        return

    print()
    tiempo_inicio_total = time.time()

    # ==============================================================================
    # PASO 1: DESCARGA DE FIFTYONE
    # ==============================================================================

    imprimir_encabezado("PASO 1/3: DESCARGA DESDE FIFTYONE (OPEN IMAGES V7)")

    print("📥 Descargando imágenes de alta calidad con detecciones")
    print("   - Gatos y Perros: CON filtros de calidad (confianza ≥60%, tamaño ≥8%)")
    print("   - Tortugas, Hormigas, Mariquitas: SIN filtros (maximizar cantidad)")
    print()

    ejecutar_script(SCRIPT_FIFTYONE, "Descarga FiftyOne")

    # Mostrar progreso
    stats_despues_fiftyone = obtener_estadisticas_dataset()
    mostrar_estadisticas("📊 DESPUÉS DE FIFTYONE", stats_despues_fiftyone)

    # ==============================================================================
    # PASO 2: DESCARGA DE KAGGLE (OPCIONAL)
    # ==============================================================================

    if kaggle_disponible:
        imprimir_encabezado("PASO 2/3: DESCARGA DESDE KAGGLE (COMPLEMENTO)")

        print("📥 Complementando con datasets de Kaggle")
        print("   Especialmente útil para clases con pocas imágenes")
        print()

        # Verificar si vale la pena ejecutar Kaggle
        necesita_kaggle = False
        for clase, count in stats_despues_fiftyone.items():
            if count < TARGET_POR_CLASE * 0.7:  # Si tiene menos del 70%
                necesita_kaggle = True
                break

        if necesita_kaggle:
            ejecutar_script(SCRIPT_KAGGLE, "Descarga Kaggle")

            # Mostrar progreso
            stats_despues_kaggle = obtener_estadisticas_dataset()
            mostrar_estadisticas("📊 DESPUÉS DE KAGGLE", stats_despues_kaggle)
        else:
            print("✅ Todas las clases tienen suficientes imágenes reales")
            print("   Saltando descarga de Kaggle")
            stats_despues_kaggle = stats_despues_fiftyone
    else:
        print("\n⏭️  PASO 2/3: KAGGLE OMITIDO (API no disponible)")
        stats_despues_kaggle = stats_despues_fiftyone

    # ==============================================================================
    # PASO 3: DATA AUGMENTATION
    # ==============================================================================

    imprimir_encabezado("PASO 3/3: DATA AUGMENTATION (COMPLETAR GAPS)")

    print("🎨 Generando imágenes sintéticas para alcanzar la meta")
    print("   Transformaciones optimizadas para realismo:")
    print("   - Rotación: ±25°")
    print("   - Zoom: ±20%")
    print("   - Brillo: ±20%")
    print("   - Sin flip vertical (animales no están boca abajo)")
    print()

    # Verificar si necesitamos augmentation
    necesita_augmentation = False
    for clase, count in stats_despues_kaggle.items():
        if count < TARGET_POR_CLASE:
            necesita_augmentation = True
            break

    if necesita_augmentation:
        ejecutar_script(SCRIPT_AUGMENTATION, "Data Augmentation")
    else:
        print("✅ Todas las clases ya alcanzaron la meta")
        print("   Saltando augmentation")

    # ==============================================================================
    # RESUMEN FINAL
    # ==============================================================================

    imprimir_encabezado("🏁 PROCESO COMPLETADO")

    # Estadísticas finales
    stats_final = obtener_estadisticas_dataset()
    mostrar_estadisticas("📊 RESULTADO FINAL", stats_final)

    # Tiempo total
    duracion_total = time.time() - tiempo_inicio_total
    horas = int(duracion_total // 3600)
    minutos = int((duracion_total % 3600) // 60)
    segundos = int(duracion_total % 60)

    print(f"⏱️  Tiempo total: {horas}h {minutos}m {segundos}s")
    print()

    # Verificar si está completo
    dataset_completo = all(count >= TARGET_POR_CLASE for count in stats_final.values())

    if dataset_completo:
        print("🎉🎉🎉 ¡DATASET PERFECTAMENTE BALANCEADO!")
        print()
        print("✅ Todas las clases tienen 30,000 imágenes")
        print(f"✅ Total: {sum(stats_final.values()):,} imágenes")
        print()
        print("=" * 80)
        print("💡 PRÓXIMO PASO: ENTRENAR EL MODELO")
        print("=" * 80)
        print()
        print("1. Abre el archivo: CNN.ipynb")
        print("2. Kernel → Restart & Run All")
        print("3. Espera el entrenamiento completo (~2-3 horas)")
        print()
        print("Configuración del modelo:")
        print("   - Batch size: 128")
        print("   - Epochs: 80 (con early stopping)")
        print("   - Learning rate: Adaptive con ReduceLROnPlateau")
        print("   - Dropout progresivo: 20% → 50%")
        print()
    else:
        print("⚠️  Dataset aún incompleto")
        print()
        clases_incompletas = [clase for clase, count in stats_final.items()
                             if count < TARGET_POR_CLASE]
        print(f"Clases que necesitan más imágenes: {', '.join(clases_incompletas)}")
        print()
        print("💡 Opciones:")
        print("   1. Ejecuta este script nuevamente para agregar más imágenes")
        print("   2. Reduce TARGET_POR_CLASE en los scripts si es difícil alcanzar 30,000")
        print("   3. Ejecuta solo el augmentation: python completar_dataset_con_augmentation.py")

    print()
    print("=" * 80)

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    try:
        generar_dataset_completo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        print("   El progreso se ha guardado. Puedes continuar ejecutando este script nuevamente.")
    except Exception as e:
        print(f"\n\n❌ ERROR INESPERADO: {str(e)}")
        import traceback
        traceback.print_exc()
