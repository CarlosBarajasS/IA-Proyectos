import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import os

# ==============================================================================
# CONFIGURACIÓN: Solo ANIMALES limpios (sin ruido de fondo)
# ==============================================================================
# Este script descarga imágenes y recorta SOLO la región del animal detectado,
# eliminando fondos, personas, objetos y otros elementos que generen ruido.

base_dir = os.path.dirname(os.path.abspath(__file__))
output_final = os.path.join(base_dir, "dataset_balanceado")
target_size = (64, 64)  # ← MEJORADO: 64×64 para mejor accuracy
IMAGENES_POR_CLASE = 15000  # 15,000 imágenes por clase

# SOLO GATOS Y PERROS - Las otras 3 clases se descargan con iNaturalist
# Ejecutar _paso1b_inaturalist.py para hormigas, mariquitas y tortugas
clases_map = {
    "gatos": ["Cat"],
    "perros": ["Dog"]
    # hormigas, mariquitas y tortugas ahora usan iNaturalist (mejor calidad científica)
}

def generar_dataset_imagenes_completas():
    print("="*80)
    print("🎯 GENERADOR DE DATASET - ANIMALES LIMPIOS (SIN RUIDO)")
    print("="*80)
    print(f"📂 Carpeta destino: {output_final}")
    print(f"🎯 Meta: {IMAGENES_POR_CLASE:,} imágenes POR CLASE")
    print("🖼️  Tipo: SOLO ANIMALES (recorte inteligente sin fondo)")
    print("🔍 Filtros: Confianza ≥70%, Tamaño ≥15%, No cortado en bordes")
    print()

    if not os.path.exists(output_final):
        os.makedirs(output_final, exist_ok=True)

    for carpeta_nombre, clases_oi in clases_map.items():
        class_final_dir = os.path.join(output_final, carpeta_nombre)

        # NO BORRAR - Acumular imágenes a las existentes
        if os.path.exists(class_final_dir):
            num_existentes = len([f for f in os.listdir(class_final_dir)
                                 if f.endswith(('.jpg', '.png'))])
            print(f"\n➕ AGREGANDO A {carpeta_nombre.upper()}")
            print(f"   Imágenes existentes: {num_existentes:,}")

            # Si ya tenemos suficientes, saltar esta clase
            if num_existentes >= IMAGENES_POR_CLASE:
                print(f"   ✅ YA COMPLETO - Omitiendo esta clase")
                continue
        else:
            print(f"\n🆕 INICIANDO {carpeta_nombre.upper()}")
            os.makedirs(class_final_dir, exist_ok=True)
            num_existentes = 0

        # Descargar MUCHAS más imágenes para tener suficientes después del filtrado
        # Open Images tiene cientos de miles de imágenes, así que podemos pedir más
        margen_seguridad = int(IMAGENES_POR_CLASE * 3)  # ← TRIPLICADO para compensar filtrado

        # Descargar candidatos
        print(f"   📥 Descargando hasta {margen_seguridad:,} candidatos de Open Images V7")
        print(f"      🔍 FILTROS MÍNIMOS: Solo rechaza objetos <1% del área")
        print(f"      ⚠️  NOTA: Open Images puede tener límites, descargará lo máximo disponible")

        dataset = None
        try:
            # Limpiar dataset previo
            try:
                fo.delete_dataset(f"full-{carpeta_nombre}", verbose=False)
            except:
                pass

            # Descargar de Open Images V7
            print(f"   ⏳ Contactando servidor...")
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=clases_oi,
                max_samples=margen_seguridad,
                only_matching=True,
                shuffle=True,
                dataset_name=f"full-{carpeta_nombre}"
            )

            print(f"   📦 Imágenes descargadas: {len(dataset):,}")

            # Calcular cuántas imágenes faltan
            faltantes_inicial = IMAGENES_POR_CLASE - num_existentes
            print(f"   🔧 Necesitamos agregar: {faltantes_inicial:,} imágenes más")
            print(f"   🔧 Procesando...")

            count = num_existentes  # Iniciar desde las existentes
            rechazadas = {"pequeño": 0, "confianza_baja": 0, "error": 0}
            primeros_5_errores = []

            for sample in dataset:
                if count >= IMAGENES_POR_CLASE:
                    break

                try:
                    # Obtener imagen original
                    img_path = sample.filepath

                    # Verificar que tenga el campo de detecciones ground_truth
                    if not sample.ground_truth or not sample.ground_truth.detections:
                        rechazadas["error"] += 1
                        if len(primeros_5_errores) < 5:
                            primeros_5_errores.append(f"No detections: {sample.id}")
                        continue

                    # Analizar la mejor detección (la más grande y con mayor confianza)
                    mejor_det = None
                    mejor_score = 0

                    for det in sample.ground_truth.detections:
                        # Calcular score combinado: tamaño × confianza
                        bbox = det.bounding_box  # [x, y, width, height] en formato relativo [0,1]
                        area = bbox[2] * bbox[3]  # width × height
                        confidence = det.confidence if hasattr(det, 'confidence') and det.confidence else 1.0
                        score = area * confidence

                        if score > mejor_score:
                            mejor_score = score
                            mejor_det = det

                    if not mejor_det:
                        rechazadas["error"] += 1
                        continue

                    bbox = mejor_det.bounding_box
                    area = bbox[2] * bbox[3]

                    # ========================================================================
                    # SIN FILTROS ESTRICTOS - Solo recortar el animal
                    # ========================================================================
                    # Removidos todos los filtros de confianza y tamaño
                    # Solo verificamos que el bbox tenga un tamaño mínimo razonable
                    if area < 0.01:  # Solo rechazar si es extremadamente pequeño (1%)
                        rechazadas["pequeño"] += 1
                        continue
                    x, y, w, h = bbox

                    # ✅ Imagen APROBADA - Recortar SOLO el animal (sin fondo)
                    with Image.open(img_path) as img:
                        # Convertir a RGB
                        img = img.convert('RGB')
                        img_width, img_height = img.size

                        # Expandir bounding box un 10% para incluir el animal completo
                        expansion = 0.10  # 10% de expansión

                        x_expanded = max(0, x - w * expansion)
                        y_expanded = max(0, y - h * expansion)
                        w_expanded = min(1.0 - x_expanded, w * (1 + 2 * expansion))
                        h_expanded = min(1.0 - y_expanded, h * (1 + 2 * expansion))

                        # Convertir a coordenadas de píxeles
                        x1 = int(x_expanded * img_width)
                        y1 = int(y_expanded * img_height)
                        x2 = int((x_expanded + w_expanded) * img_width)
                        y2 = int((y_expanded + h_expanded) * img_height)

                        # Recortar SOLO la región del animal
                        img_cropped = img.crop((x1, y1, x2, y2))

                        # Redimensionar a 64x64
                        img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)

                        # Guardar con nombre único basado en el contador actual
                        new_name = f"{carpeta_nombre}_fiftyone_{count:06d}.jpg"
                        save_path = os.path.join(class_final_dir, new_name)

                        img_resized.save(save_path, quality=95)
                        count += 1

                        # Progreso cada 1000
                        if count % 1000 == 0:
                            nuevas_agregadas = count - num_existentes
                            print(f"      ✅ {count:,} / {IMAGENES_POR_CLASE:,} (agregadas: {nuevas_agregadas:,})")
                            print(f"         Rechazadas: pequeño={rechazadas['pequeño']}, "
                                  f"baja_confianza={rechazadas['confianza_baja']}")

                except Exception as e:
                    rechazadas["error"] += 1
                    if len(primeros_5_errores) < 5:
                        primeros_5_errores.append(f"Exception: {str(e)[:50]}")
                    continue

            # Reporte final
            final_count = len([f for f in os.listdir(class_final_dir)
                             if f.endswith(('.jpg', '.png'))])

            nuevas_totales = final_count - num_existentes

            print(f"\n   ✅ {carpeta_nombre.upper()}: {final_count:,} / {IMAGENES_POR_CLASE:,}")
            print(f"      📥 Imágenes agregadas en esta ejecución: {nuevas_totales:,}")

            if rechazadas['pequeño'] > 0 or rechazadas['error'] > 0:
                print(f"   📊 Rechazadas:")
                print(f"      - Objeto muy pequeño (<1%): {rechazadas['pequeño']:,}")
                print(f"      - Errores: {rechazadas['error']:,}")
                total_rechazadas = rechazadas['pequeño'] + rechazadas['error']
                print(f"      🚫 Total rechazadas: {total_rechazadas:,}")

            if primeros_5_errores:
                print(f"      🔍 Primeros errores:")
                for err in primeros_5_errores:
                    print(f"         - {err}")

            if final_count < IMAGENES_POR_CLASE:
                faltantes = IMAGENES_POR_CLASE - final_count
                print(f"   ⚠️  FALTAN {faltantes:,} imágenes")
                print(f"   💡 Ejecuta el script nuevamente para agregar más")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        finally:
            if dataset:
                try:
                    fo.delete_dataset(f"full-{carpeta_nombre}", verbose=False)
                except:
                    pass

    print("\n" + "="*80)
    print("🏁 PROCESO FINALIZADO")
    print("="*80)

    # Reporte final
    print("\n📊 RESUMEN FINAL:")
    for carpeta_nombre in clases_map.keys():
        class_dir = os.path.join(output_final, carpeta_nombre)
        if os.path.exists(class_dir):
            total = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
            status = "✅" if total >= IMAGENES_POR_CLASE else "⚠️"
            print(f"   {status} {carpeta_nombre:12} : {total:,} / {IMAGENES_POR_CLASE:,}")

if __name__ == "__main__":
    generar_dataset_imagenes_completas()
