import os
from collections import defaultdict

import cv2 as cv
import numpy as np

# Rutas relativas a este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "videos", "recortes")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_facial_lbph.xml")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_dict.npy")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
TARGET_SIZE = (100, 100)


def load_images_from_directory(directory, label_id, faces, labels):
    """Carga imágenes en escala de grises desde un directorio y las añade al dataset."""
    added_any = False
    for entry in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, entry)
        if not (
            os.path.isfile(img_path)
            and entry.lower().endswith(IMAGE_EXTENSIONS)
        ):
            continue

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Advertencia: no se pudo leer la imagen {img_path}; se omite.")
            continue

        img = cv.resize(img, TARGET_SIZE)
        img = cv.equalizeHist(img)
        faces.append(img)
        labels.append(label_id)
        added_any = True

    if not added_any:
        print(f"Advertencia: no se encontraron imágenes válidas en {directory}.")

    return added_any


def gather_dataset():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"No se encontró el directorio de datos: {DATASET_DIR}")

    faces = []
    labels = []
    label_dict = {}
    samples_per_label = defaultdict(int)

    entries = sorted(os.listdir(DATASET_DIR))
    subdirectories = [
        entry for entry in entries if os.path.isdir(os.path.join(DATASET_DIR, entry))
    ]

    if subdirectories:
        current_label = 0
        for person_name in subdirectories:
            person_path = os.path.join(DATASET_DIR, person_name)
            if load_images_from_directory(person_path, current_label, faces, labels):
                label_dict[current_label] = person_name
                samples_per_label[person_name] = labels.count(current_label)
                current_label += 1
    else:
        # Sin subdirectorios: tratar todas las imágenes como un único sujeto
        label_name = os.path.basename(DATASET_DIR.rstrip(os.sep))
        if load_images_from_directory(DATASET_DIR, 0, faces, labels):
            label_dict[0] = label_name
            samples_per_label[label_name] = len(labels)

    return faces, labels, label_dict, samples_per_label


def require_minimum_samples(samples_per_label, minimum=5):
    lacking = {name: count for name, count in samples_per_label.items() if count < minimum}
    if lacking:
        details = ", ".join(f"{name}: {count}" for name, count in lacking.items())
        raise ValueError(
            "Cada persona necesita al menos "
            f"{minimum} imágenes. Recolecta más datos antes de entrenar. "
            f"Faltantes -> {details}"
        )


def main():
    faces, labels, label_dict, samples_per_label = gather_dataset()

    if len(faces) < 2:
        raise ValueError(
            "Se necesitan al menos dos imágenes válidas para entrenar el modelo. "
            f"Revisa el contenido de {DATASET_DIR}."
        )

    require_minimum_samples(samples_per_label)

    faces_array = np.array(faces)
    labels_array = np.array(labels, dtype=np.int32)

    recognizer = cv.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    recognizer.train(faces_array, labels_array)
    recognizer.save(MODEL_PATH)

    np.save(LABEL_MAP_PATH, label_dict)

    print(f"Modelo entrenado y guardado en: {MODEL_PATH}")
    print("Personas reconocibles y cantidad de muestras por etiqueta:")
    for label_id, name in label_dict.items():
        print(f" - {name}: {samples_per_label[name]} muestras")


if __name__ == "__main__":
    main()
