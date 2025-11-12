import cv2
import os

# Ruta del dataset original
input_dir = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\dataset"
# Ruta donde se guardarán las imágenes redimensionadas
output_dir = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\DATASET"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorremos todas las subcarpetas (una por clase)
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                resized = cv2.resize(img, (64, 64))
                cv2.imwrite(os.path.join(output_folder, filename), resized)

print("✅ Todas las imágenes fueron redimensionadas a 64x64 y guardadas en", output_dir)
