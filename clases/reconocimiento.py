import os

import cv2 as cv
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_facial_lbph.xml")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_dict.npy")

CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_alt2.xml"
THRESHOLD = 65.0  # Umbral de confianza para LBPH (menor es mejor)
TARGET_SIZE = (100, 100)


def build_capture_source():
    source = input("Ruta del video (deja en blanco para usar la cámara): ").strip()
    if not source:
        return 0
    if source.isdigit():
        return int(source)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"No se encontró el archivo de video: {source}")
    return source


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado ({MODEL_PATH}). "
            "Ejecuta primero entrenamiento.py."
        )

    if not os.path.isfile(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"No se encontró el mapa de etiquetas ({LABEL_MAP_PATH}). "
            "Ejecuta primero entrenamiento.py."
        )

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_dict = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    return recognizer, label_dict


def main():
    recognizer, label_dict = load_model()

    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    cap = cv.VideoCapture(build_capture_source())

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video/cámara.")

    print("\nControles:")
    print(" - ESC: salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o pérdida de señal.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv.resize(face_roi, TARGET_SIZE)

            label_id, confidence = recognizer.predict(face_roi)

            if confidence <= THRESHOLD:
                name = label_dict.get(label_id, "Desconocido")
                color = (0, 255, 0)
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Desconocido"
                color = (0, 0, 255)
                text = f"{name} ({confidence:.1f})"

            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(
                frame,
                text,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv.imshow("Reconocimiento facial", frame)

        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
