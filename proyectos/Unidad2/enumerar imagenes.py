import os

# Ruta de la carpeta donde están las imágenes
carpeta = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\mas mariquitas"

# Extensiones válidas (puedes agregar más)
extensiones = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

# Lista de archivos en la carpeta
archivos = os.listdir(carpeta)

contador = 2060
for archivo in archivos:
    nombre, extension = os.path.splitext(archivo)
    if extension.lower() in extensiones:
        nuevo_nombre = f"imagen_{contador}{extension.lower()}"
        ruta_vieja = os.path.join(carpeta, archivo)
        ruta_nueva = os.path.join(carpeta, nuevo_nombre)
        os.rename(ruta_vieja, ruta_nueva)
        print(f"✅ {archivo} → {nuevo_nombre}")
        contador += 1

print("\n🎉 Renombrado completo. Total de imágenes:", contador)
