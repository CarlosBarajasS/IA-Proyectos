import os
import shutil

# Ruta principal donde están las carpetas con imágenes
carpeta_principal = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\MARIQUITAS"

# Carpeta donde se guardarán todas las imágenes unificadas
carpeta_destino = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\dataset\mariquitas"

# Crear carpeta destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# Extensiones de imagen válidas
extensiones = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

contador = 0

# Recorrer todas las subcarpetas
for carpeta_raiz, subcarpetas, archivos in os.walk(carpeta_principal):
    for archivo in archivos:
        if archivo.lower().endswith(extensiones):
            ruta_origen = os.path.join(carpeta_raiz, archivo)
            ruta_destino = os.path.join(carpeta_destino, archivo)

            # Evitar sobrescribir si hay imágenes con el mismo nombre
            if os.path.exists(ruta_destino):
                nombre, extension = os.path.splitext(archivo)
                nuevo_nombre = f"{nombre}_{contador}{extension}"
                ruta_destino = os.path.join(carpeta_destino, nuevo_nombre)

            shutil.move(ruta_origen, ruta_destino)
            contador += 1
            print(f"Copiada: {ruta_origen} → {ruta_destino}")

print(f"\n✅ Se copiaron {contador} imágenes a {carpeta_destino}")
