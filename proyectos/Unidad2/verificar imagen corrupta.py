import cv2
import numpy as np
import os

images = []
labels = []

# 🔹 RUTA CORREGIDA con raw string
base_path = r"C:\Users\Adolfo\Documents\ProyectsVisualStudioCode\Inteligencia Artificial\Entorno Virtual\Proyecto Clasificar Animales - copia\dataset"

for label in os.listdir(base_path):
    path = os.path.join(base_path, label)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Imagen corrupta o no legible: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)

X = np.array(images, dtype=np.uint8)
y = np.array(labels)

print("✅ Carga completada.")
print(f"Total imágenes: {len(X)}")
print(f"Clases encontradas: {set(y)}")
