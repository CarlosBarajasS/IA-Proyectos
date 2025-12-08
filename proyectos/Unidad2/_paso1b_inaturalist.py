"""
DESCARGADOR DESDE iNATURALIST - DATOS CIENTÍFICOS DE CALIDAD
Descarga imágenes de hormigas, mariquitas y tortugas con clasificación taxonómica verificada
Fuente: iNaturalist Research-Grade Observations

VENTAJAS:
- Clasificación taxonómica exacta (Formicidae, Coccinellidae, Testudines)
- Fotos verificadas por científicos (Research Grade)
- Licencias educativas (CC0, CC-BY)
- Alta calidad y diversidad
"""

import os
import sys
import requests
from io import BytesIO
from PIL import Image
import time

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
TARGET_POR_CLASE = 15000  # Objetivo: 15,000 imágenes por clase

# Mapeo de taxones de iNaturalist (IDs científicos)
TAXONES_INATURALIST = {
    "hormigas": {
        "taxon_id": 47336,  # Formicidae (familia de hormigas)
        "nombre_cientifico": "Formicidae",
        "nombre_comun": "Ants / Hormigas"
    },
    "mariquitas": {
        "taxon_id": 47744,  # Coccinellidae (familia de mariquitas)
        "nombre_cientifico": "Coccinellidae",
        "nombre_comun": "Lady Beetles / Mariquitas"
    },
    "tortugas": {
        "taxon_id": 39532,  # Testudines (orden de tortugas)
        "nombre_cientifico": "Testudines",
        "nombre_comun": "Turtles / Tortugas"
    }
}

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def obtener_observaciones_inaturalist(taxon_id, page=1, per_page=200):
    """
    Obtiene observaciones de iNaturalist usando su API REST

    Parámetros de búsqueda:
    - taxon_id: ID del taxón (familia/orden)
    - quality_grade: research (verificadas por científicos)
    - photos: true (solo con fotos)
    - license: cc0,cc-by,cc-by-nc (licencias abiertas)
    - per_page: 200 (máximo permitido)
    """
    url = "https://api.inaturalist.org/v1/observations"

    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",  # Solo observaciones verificadas
        "photos": "true",
        "license": "cc0,cc-by,cc-by-nc",  # Licencias educativas
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  Error en API: {str(e)}")
        return None

def descargar_imagen(url, clase, contador, output_dir, max_reintentos=3):
    """
    Descarga y procesa una imagen de iNaturalist

    Características:
    - Convierte a RGB
    - Redimensiona a 64×64 con LANCZOS (alta calidad)
    - Guarda como JPEG con calidad 95
    - Reintentos automáticos en caso de error
    """
    carpeta_destino = os.path.join(output_dir, clase)
    os.makedirs(carpeta_destino, exist_ok=True)

    nombre_archivo = f"{clase}_inat_{contador:06d}.jpg"
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)

    # Si ya existe, omitir
    if os.path.exists(ruta_completa):
        return True

    for intento in range(max_reintentos):
        try:
            # Descargar imagen
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            # Procesar imagen
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')

            # Redimensionar a 64×64 (mejorado para mejor accuracy)
            img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)

            # Guardar
            img_resized.save(ruta_completa, "JPEG", quality=95, optimize=True)

            return True

        except Exception as e:
            if intento < max_reintentos - 1:
                time.sleep(1)  # Esperar antes de reintentar
                continue
            else:
                return False

    return False

def contar_imagenes_existentes(clase, output_dir):
    """Cuenta imágenes ya descargadas de una clase"""
    carpeta = os.path.join(output_dir, clase)
    if not os.path.exists(carpeta):
        return 0

    archivos = [f for f in os.listdir(carpeta) if f.startswith(f"{clase}_inat_") and f.endswith('.jpg')]
    return len(archivos)

# ==============================================================================
# DESCARGA PRINCIPAL
# ==============================================================================

def descargar_clase_inaturalist(clase, config):
    """
    Descarga imágenes de una clase desde iNaturalist

    Proceso:
    1. Verificar cuántas imágenes ya existen
    2. Calcular cuántas faltan
    3. Paginar la API hasta completar el objetivo
    4. Descargar y procesar cada foto
    """
    print(f"\n{'='*80}")
    print(f"🔬 DESCARGANDO: {config['nombre_comun']}")
    print(f"   Taxón: {config['nombre_cientifico']} (ID: {config['taxon_id']})")
    print(f"{'='*80}\n")

    # Contar existentes
    existentes = contar_imagenes_existentes(clase, output_dir)
    print(f"   📊 Imágenes existentes: {existentes:,}")

    if existentes >= TARGET_POR_CLASE:
        print(f"   ✅ Clase ya completa ({existentes:,} / {TARGET_POR_CLASE:,})")
        return existentes

    faltantes = TARGET_POR_CLASE - existentes
    print(f"   🎯 Objetivo: {TARGET_POR_CLASE:,}")
    print(f"   ⏳ Faltan: {faltantes:,}")
    print()

    # Empezar descarga
    contador = existentes
    page = 1
    sin_resultados_consecutivos = 0

    while contador < TARGET_POR_CLASE:
        print(f"   📄 Página {page}...", end=" ", flush=True)

        # Obtener observaciones
        data = obtener_observaciones_inaturalist(config['taxon_id'], page=page)

        if not data or 'results' not in data or len(data['results']) == 0:
            print("Sin resultados")
            sin_resultados_consecutivos += 1

            if sin_resultados_consecutivos >= 3:
                print(f"\n   ⚠️  No hay más imágenes disponibles en iNaturalist")
                break

            page += 1
            continue

        sin_resultados_consecutivos = 0
        resultados = data['results']
        print(f"{len(resultados)} observaciones")

        descargadas_pagina = 0
        errores_pagina = 0

        for obs in resultados:
            if contador >= TARGET_POR_CLASE:
                break

            # Obtener fotos de la observación
            photos = obs.get('photos', [])

            for photo in photos:
                if contador >= TARGET_POR_CLASE:
                    break

                # Obtener URL de tamaño mediano (mejor calidad)
                url = photo.get('url')
                if url:
                    # Cambiar a tamaño 'medium' (500px) para mejor calidad
                    url = url.replace('/square.', '/medium.')

                    if descargar_imagen(url, clase, contador, output_dir):
                        contador += 1
                        descargadas_pagina += 1

                        if contador % 100 == 0:
                            progreso = (contador / TARGET_POR_CLASE * 100)
                            print(f"      ✓ {contador:,} / {TARGET_POR_CLASE:,} ({progreso:.1f}%)")
                    else:
                        errores_pagina += 1

        print(f"      ✓ Descargadas: {descargadas_pagina} | Errores: {errores_pagina}")

        page += 1

        # Rate limiting: esperar 1 segundo entre páginas
        time.sleep(1)

    print()
    print(f"   ✅ Descarga completada: {contador:,} imágenes")

    return contador

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("="*80)
    print("🔬 DESCARGADOR iNATURALIST - DATOS CIENTÍFICOS DE CALIDAD")
    print("="*80)
    print()
    print("📚 FUENTE: iNaturalist Research-Grade Observations")
    print("   - Solo observaciones verificadas por científicos")
    print("   - Clasificación taxonómica exacta")
    print("   - Licencias educativas (CC0, CC-BY, CC-BY-NC)")
    print()
    print(f"🎯 OBJETIVO: {TARGET_POR_CLASE:,} imágenes por clase")
    print(f"📐 DIMENSIONES: 64×64 píxeles (RGB)")
    print()

    # Crear carpeta de salida
    os.makedirs(output_dir, exist_ok=True)

    # Clases a descargar (solo las problemáticas)
    clases_descargar = ["hormigas", "mariquitas", "tortugas"]

    print("🐜🐞🐢 CLASES A DESCARGAR:")
    for clase in clases_descargar:
        config = TAXONES_INATURALIST[clase]
        print(f"   - {clase.capitalize()}: {config['nombre_cientifico']} (Taxon ID: {config['taxon_id']})")
    print()

    # Confirmar con usuario
    respuesta = input("¿Deseas iniciar la descarga desde iNaturalist? (s/n): ").strip().lower()

    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Descarga cancelada")
        return

    print()
    print("="*80)
    print("🚀 INICIANDO DESCARGA...")
    print("="*80)
    print()
    print("⏳ TIEMPO ESTIMADO: 30-90 minutos (depende de la conexión)")
    print("   iNaturalist tiene rate limiting, respetamos 1 segundo entre páginas")
    print()

    inicio = time.time()
    resultados = {}

    # Descargar cada clase
    for clase in clases_descargar:
        config = TAXONES_INATURALIST[clase]
        total_descargadas = descargar_clase_inaturalist(clase, config)
        resultados[clase] = total_descargadas

    fin = time.time()
    duracion = fin - inicio

    # Resumen final
    print()
    print("="*80)
    print("📊 RESUMEN FINAL - iNATURALIST")
    print("="*80)
    print()

    total_imagenes = 0
    for clase in clases_descargar:
        total = resultados.get(clase, 0)
        total_imagenes += total
        pct_completo = (total / TARGET_POR_CLASE * 100) if TARGET_POR_CLASE > 0 else 0

        status = "✅" if total >= TARGET_POR_CLASE else "⚠️"
        print(f"{status} {clase.capitalize():12} : {total:,} / {TARGET_POR_CLASE:,} ({pct_completo:.1f}%)")

    print()
    print(f"📊 Total descargado: {total_imagenes:,} imágenes")
    print(f"⏱️  Tiempo total: {duracion/60:.1f} minutos")
    print()

    # Próximos pasos
    print("="*80)
    print("💡 PRÓXIMOS PASOS:")
    print("="*80)
    print()

    clases_incompletas = [c for c in clases_descargar if resultados.get(c, 0) < TARGET_POR_CLASE]

    if len(clases_incompletas) > 0:
        print("⚠️  Algunas clases no alcanzaron 15,000 imágenes:")
        for clase in clases_incompletas:
            faltantes = TARGET_POR_CLASE - resultados.get(clase, 0)
            print(f"   - {clase}: Faltan {faltantes:,}")
        print()
        print("OPCIONES:")
        print("1. Complementar con FiftyOne:")
        print("   python _paso1_fiftyone.py")
        print()
        print("2. Complementar con Kaggle:")
        print("   python _paso2_kaggle.py")
        print()
        print("3. Completar con augmentation:")
        print("   python _paso3_augmentation.py")
        print()
    else:
        print("✅ ¡TODAS LAS CLASES COMPLETAS CON IMÁGENES DE iNATURALIST!")
        print()
        print("Ahora descarga gatos y perros:")
        print("   python _paso1_fiftyone.py")
        print()

    print("="*80)

if __name__ == "__main__":
    main()
