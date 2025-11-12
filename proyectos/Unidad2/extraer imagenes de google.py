# Instala la librería si no la tienes:
# pip install icrawler

from icrawler.builtin import BingImageCrawler
import os

# ---------------------------
# CONFIGURACIÓN DEL DATASET
# ---------------------------

# Carpeta donde se guardarán todas las imágenes
dataset_dir = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\mas mariquitas"  # Cambia esto a tu ruta

# Lista de clases que quieres descargar
clases = ["tortugas"]

# Número máximo de imágenes por clase
imagenes_por_clase = 1000

# Tamaño mínimo de las imágenes (opcional)
min_width = 30
min_height = 30


for clase in clases:
    print(f"\n⬇️ Descargando imágenes para la clase: {clase}")

    # Crear carpeta para la clase si no existe
    clase_dir = os.path.join(dataset_dir, clase)
    os.makedirs(clase_dir, exist_ok=True)

    # Crear el crawler de Bing
    crawler = BingImageCrawler(storage={"root_dir": clase_dir})

    # Descargar imágenes
    crawler.crawl(
        keyword=clase,
        max_num=imagenes_por_clase,
        min_size=(min_width, min_height),
        file_idx_offset=0  # Para que no sobrescriba archivos si se ejecuta varias veces
    )

print("\n✅ Descarga completa para todas las clases.")
