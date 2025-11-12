import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Carpeta con tus imágenes originales (la clase que tiene pocas)
input_dir = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\dataset\mariquitas"

# Carpeta donde se guardarán las imágenes generadas
output_dir = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\mas mariquitas"
os.makedirs(output_dir, exist_ok=True)

# Parámetros del aumento
imagenes_a_generar = 10000

# Crear el generador de aumento
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Cargar imágenes base
imagenes_base = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
num_base = len(imagenes_base)
print(f"📸 Imágenes originales encontradas: {num_base}")

# Cuántas nuevas debe generar por imagen base
aumentos_por_imagen = max(1, imagenes_a_generar // num_base)
print(f"🔁 Se generarán {aumentos_por_imagen} nuevas por imagen base aproximadamente")

contador_total = 0

for img_name in imagenes_base:
    img_path = os.path.join(input_dir, img_name)
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_dir,
                              save_prefix="aug",
                              save_format="jpg"):
        i += 1
        contador_total += 1
        if i >= aumentos_por_imagen or contador_total >= imagenes_a_generar:
            break
    if contador_total >= imagenes_a_generar:
        break

print(f"✅ Se generaron {contador_total} imágenes nuevas en '{output_dir}'")