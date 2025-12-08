"""
GENERADOR MAESTRO DE DATASET - ESTRATEGIA HÍBRIDA OPTIMIZADA
Orquesta la descarga desde múltiples fuentes para maximizar calidad

ESTRATEGIA:
1. iNaturalist (Research-Grade) → Hormigas, Mariquitas, Tortugas
   - Datos científicos verificados
   - Clasificación taxonómica exacta
   - ~12,000-15,000 imágenes por clase

2. FiftyOne (Open Images V7) → Gatos, Perros
   - Dataset de Google con bounding boxes
   - ~15,000 imágenes por clase

3. Kaggle (opcional) → Complementar si faltan
   - Datasets públicos variados

4. Augmentation → Completar hasta 15,000
   - Solo si no alcanzamos con imágenes reales
   - Máximo 20-30% sintéticas

RESULTADO ESPERADO:
- 75,000 imágenes totales (15,000 por clase)
- 70-90% imágenes reales
- Accuracy estimado: 70-80%
"""

import os
import sys
import subprocess

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset_balanceado")
TARGET = 15000

def contar_imagenes():
    """Cuenta imágenes por clase"""
    stats = {}
    clases = ["gatos", "perros", "hormigas", "mariquitas", "tortugas"]

    for clase in clases:
        carpeta = os.path.join(dataset_dir, clase)
        if not os.path.exists(carpeta):
            stats[clase] = 0
            continue

        archivos = [f for f in os.listdir(carpeta) if f.endswith(('.jpg', '.png'))]
        stats[clase] = len(archivos)

    return stats

def mostrar_resumen(titulo, stats):
    """Muestra resumen del estado del dataset"""
    print()
    print("="*80)
    print(f"📊 {titulo}")
    print("="*80)
    print()

    total = 0
    for clase in ["gatos", "perros", "hormigas", "mariquitas", "tortugas"]:
        count = stats.get(clase, 0)
        total += count
        pct = (count / TARGET * 100) if TARGET > 0 else 0

        status = "✅" if count >= TARGET else "⚠️" if count > 0 else "❌"
        faltantes = max(0, TARGET - count)

        print(f"{status} {clase.capitalize():12} : {count:,} / {TARGET:,} ({pct:.1f}%) - Faltan: {faltantes:,}")

    print("-"*80)
    print(f"{'TOTAL':12} : {total:,} / {TARGET * 5:,}")
    print()

def ejecutar_script(nombre_script, descripcion):
    """Ejecuta un script de Python y maneja errores"""
    print()
    print("="*80)
    print(f"🚀 {descripcion}")
    print("="*80)
    print()

    script_path = os.path.join(base_dir, nombre_script)

    if not os.path.exists(script_path):
        print(f"❌ Error: No se encontró {nombre_script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=base_dir,
            check=False
        )

        if result.returncode == 0:
            print(f"\n✅ {descripcion} - COMPLETADO")
            return True
        else:
            print(f"\n⚠️  {descripcion} - Finalizado con código {result.returncode}")
            return False

    except KeyboardInterrupt:
        print(f"\n⏸️  {descripcion} - INTERRUMPIDO POR USUARIO")
        return False
    except Exception as e:
        print(f"\n❌ Error al ejecutar {nombre_script}: {str(e)}")
        return False

def main():
    print("="*80)
    print("🎯 GENERADOR MAESTRO DE DATASET - ESTRATEGIA HÍBRIDA OPTIMIZADA")
    print("="*80)
    print()
    print("📚 FUENTES DE DATOS:")
    print("   1. iNaturalist → Hormigas, Mariquitas, Tortugas (datos científicos)")
    print("   2. FiftyOne → Gatos, Perros (Open Images V7)")
    print("   3. Kaggle → Complementar si faltan (opcional)")
    print("   4. Augmentation → Completar hasta 15,000 (solo si es necesario)")
    print()
    print(f"🎯 OBJETIVO: {TARGET:,} imágenes por clase × 5 clases = {TARGET * 5:,} total")
    print(f"📐 DIMENSIONES: 32×32 píxeles (RGB)")
    print()

    # Estado inicial
    stats_inicial = contar_imagenes()
    mostrar_resumen("ESTADO INICIAL", stats_inicial)

    # Confirmar inicio
    print("="*80)
    print("💡 PLAN DE EJECUCIÓN:")
    print("="*80)
    print()
    print("PASO 1: iNaturalist (30-90 min)")
    print("   → Descarga hormigas, mariquitas, tortugas")
    print("   → ~12,000-15,000 imágenes por clase")
    print()
    print("PASO 2: FiftyOne (15-30 min)")
    print("   → Descarga gatos y perros")
    print("   → ~15,000 imágenes por clase")
    print()
    print("PASO 3: Kaggle (OPCIONAL, 10-20 min)")
    print("   → Solo si faltan imágenes")
    print()
    print("PASO 4: Augmentation (10-15 min)")
    print("   → Completar hasta 15,000 con imágenes sintéticas")
    print()
    print("⏱️  TIEMPO TOTAL ESTIMADO: 1-2 horas")
    print()

    respuesta = input("¿Deseas iniciar el proceso completo? (s/n): ").strip().lower()

    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Proceso cancelado")
        return

    # =========================================================================
    # PASO 1: iNaturalist (Hormigas, Mariquitas, Tortugas)
    # =========================================================================

    print("\n" + "="*80)
    print("PASO 1/4: DESCARGA DESDE iNATURALIST")
    print("="*80)

    ejecutar_script("_paso1b_inaturalist.py", "Descargando desde iNaturalist")

    stats_post_inat = contar_imagenes()
    mostrar_resumen("DESPUÉS DE iNATURALIST", stats_post_inat)

    # =========================================================================
    # PASO 2: FiftyOne (Gatos, Perros)
    # =========================================================================

    print("\n" + "="*80)
    print("PASO 2/4: DESCARGA DESDE FIFTYONE (OPEN IMAGES V7)")
    print("="*80)

    ejecutar_script("_paso1_fiftyone.py", "Descargando desde FiftyOne")

    stats_post_fiftyone = contar_imagenes()
    mostrar_resumen("DESPUÉS DE FIFTYONE", stats_post_fiftyone)

    # =========================================================================
    # PASO 3: Kaggle (Opcional - Solo si faltan)
    # =========================================================================

    clases_incompletas = [c for c, count in stats_post_fiftyone.items() if count < TARGET]

    if len(clases_incompletas) > 0:
        print("\n" + "="*80)
        print("PASO 3/4: DESCARGA DESDE KAGGLE (COMPLEMENTAR)")
        print("="*80)
        print()
        print("⚠️  Las siguientes clases no alcanzaron 15,000 imágenes:")
        for clase in clases_incompletas:
            faltantes = TARGET - stats_post_fiftyone.get(clase, 0)
            print(f"   - {clase}: Faltan {faltantes:,}")
        print()

        respuesta = input("¿Deseas intentar completar con Kaggle? (s/n): ").strip().lower()

        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            ejecutar_script("_paso2_kaggle.py", "Descargando desde Kaggle")

            stats_post_kaggle = contar_imagenes()
            mostrar_resumen("DESPUÉS DE KAGGLE", stats_post_kaggle)
        else:
            print("⏭️  Kaggle omitido")
            stats_post_kaggle = stats_post_fiftyone
    else:
        print("\n✅ TODAS LAS CLASES COMPLETAS CON IMÁGENES REALES - OMITIENDO KAGGLE")
        stats_post_kaggle = stats_post_fiftyone

    # =========================================================================
    # PASO 4: Augmentation (Completar hasta 15,000)
    # =========================================================================

    clases_necesitan_augmentation = [c for c, count in stats_post_kaggle.items() if count < TARGET]

    if len(clases_necesitan_augmentation) > 0:
        print("\n" + "="*80)
        print("PASO 4/4: DATA AUGMENTATION (COMPLETAR HASTA 15,000)")
        print("="*80)
        print()
        print("⚠️  Las siguientes clases necesitan augmentation:")
        for clase in clases_necesitan_augmentation:
            actuales = stats_post_kaggle.get(clase, 0)
            faltantes = TARGET - actuales
            pct_real = (actuales / TARGET * 100) if TARGET > 0 else 0
            print(f"   - {clase}: {actuales:,} reales → Generar {faltantes:,} sintéticas ({pct_real:.1f}% real)")
        print()

        respuesta = input("¿Deseas ejecutar data augmentation? (s/n): ").strip().lower()

        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            ejecutar_script("_paso3_augmentation.py", "Generando imágenes sintéticas")

            stats_final = contar_imagenes()
            mostrar_resumen("DESPUÉS DE AUGMENTATION", stats_final)
        else:
            print("⏭️  Augmentation omitido")
            stats_final = stats_post_kaggle
    else:
        print("\n🎉 ¡TODAS LAS CLASES COMPLETAS SIN NECESIDAD DE AUGMENTATION!")
        stats_final = stats_post_kaggle

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print()
    print("="*80)
    print("🏁 RESUMEN FINAL - DATASET COMPLETO")
    print("="*80)
    print()

    total_imagenes = sum(stats_final.values())
    clases_completas = len([c for c in stats_final.values() if c >= TARGET])

    print(f"📊 Total de imágenes: {total_imagenes:,} / {TARGET * 5:,}")
    print(f"✅ Clases completas: {clases_completas} / 5")
    print()

    print("DETALLE POR CLASE:")
    for clase in ["gatos", "perros", "hormigas", "mariquitas", "tortugas"]:
        count = stats_final.get(clase, 0)
        pct = (count / TARGET * 100) if TARGET > 0 else 0
        status = "✅" if count >= TARGET else "⚠️"
        print(f"{status} {clase.capitalize():12} : {count:,} ({pct:.1f}%)")

    print()

    # Estimación de accuracy
    if total_imagenes >= TARGET * 5:
        print("🎉 ¡DATASET COMPLETO Y BALANCEADO!")
        print()
        print("📈 PREDICCIÓN:")
        print("   - Calidad: EXCELENTE (alta proporción de imágenes reales)")
        print("   - Accuracy estimado: 70-80%")
        print()
        print("💡 PRÓXIMO PASO: ENTRENAR EL MODELO")
        print()
        print("1. Abre CNN.ipynb")
        print("2. Selecciona kernel: Python 3.11.9 (.venv_keras2)")
        print("3. Restart Kernel")
        print("4. Run All Cells")
        print()
    else:
        print("⚠️  Dataset incompleto")
        print()
        print("Considera:")
        print("1. Ejecutar augmentation para completar")
        print("2. Reducir TARGET_POR_CLASE a 10,000 en los scripts")
        print()

    print("="*80)

if __name__ == "__main__":
    main()
