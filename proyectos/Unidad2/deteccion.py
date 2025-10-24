import os
import time

import cv2 as cv

# Directorios y configuraciones base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "videos", "recortes")
os.makedirs(DATASET_DIR, exist_ok=True)

CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_alt2.xml"
face_detector = cv.CascadeClassifier(CASCADE_PATH)

CAPTURE_EVERY_N_FRAMES = 3  # Capturar una imagen cada N cuadros detectados
RESIZE_TO = (100, 100)


def ask_input(prompt, default=None, cast_fn=None):
    """Solicita un valor por consola y aplica conversión si es necesario."""
    value = input(prompt).strip()
    if not value:
        return default
    if cast_fn:
        try:
            return cast_fn(value)
        except Exception:
            print("Valor inválido, se usará el valor por defecto.")
            return default
    return value


def build_capture_source():
    """Determina si se usará la cámara o un archivo de video."""
    source = input(
        "Ruta del video (deja en blanco para usar la cámara por defecto): "
    ).strip()

    if not source:
        return 0

    if source.isdigit():
        return int(source)

    if not os.path.isfile(source):
        raise FileNotFoundError(f"No se encontró el archivo de video: {source}")
    return source


def capture_faces():
    label = ask_input("Nombre/etiqueta de la persona: ", default="persona")
    label = label.replace(" ", "_")

    num_images = ask_input(
        "Número de imágenes a capturar (50 por defecto): ",
        default=50,
        cast_fn=int,
    )
    num_images = max(1, num_images)

    destination_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(destination_dir, exist_ok=True)

    existing_files = [
        name for name in os.listdir(destination_dir) if name.lower().endswith((".png", ".jpg"))
    ]
    next_index = len(existing_files)

    source = build_capture_source()
    cap = cv.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video/cámara.")

    saved = 0
    frame_counter = 0
    last_capture_time = 0.0
    min_seconds_between_captures = 0.2

    print("\nControles:")
    print(" - ESC: salir")
    print(" - ESPACIO: pausar/reanudar captura automática\n")

    auto_capture = True

    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o pérdida de señal. Se detiene la captura.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv.resize(face_roi, RESIZE_TO)

            should_capture = (
                auto_capture
                and frame_counter % CAPTURE_EVERY_N_FRAMES == 0
                and time.time() - last_capture_time >= min_seconds_between_captures
            )

            if should_capture:
                filename = f"{label}_{next_index + saved:04}.png"
                save_path = os.path.join(destination_dir, filename)
                cv.imwrite(save_path, face_roi)
                last_capture_time = time.time()
                saved += 1

            cv.imshow("Rostro detectado", face_roi)

        status_text = f"Etiqueta: {label} | Guardadas: {saved}/{num_images} | Auto: {'ON' if auto_capture else 'OFF'}"
        cv.putText(
            frame,
            status_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv.imshow("Captura de rostros", frame)

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Captura interrumpida por el usuario.")
            break
        if key == 32:  # Espacio
            auto_capture = not auto_capture
            time.sleep(0.2)

        frame_counter += 1

    cap.release()
    cv.destroyAllWindows()
    print(f"Se guardaron {saved} imágenes en {destination_dir}")


if __name__ == "__main__":
    capture_faces()
