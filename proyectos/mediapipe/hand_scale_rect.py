import math
from pathlib import Path

import cv2
import mediapipe as mp


def main():
    """
    Use MediaPipe Hands to scale an on-screen rectangle based on the distance
    between the thumb and index finger tips.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara web.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    base_width = 150  # tamaño por defecto cuando la mano no esté visible
    min_scale, max_scale = 0.5, 2.2  # límites para el escalado del rectángulo

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espejo para interacción natural
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)
        rect_width = base_width

        if result.multi_hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

            distance = math.hypot(x2 - x1, y2 - y1)
            normalized = max(min(distance / 120.0, max_scale), min_scale)
            rect_width = int(base_width * normalized)

        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        half_width = rect_width // 2
        half_height = int(rect_width * 0.6) // 2

        top_left = (cx - half_width, cy - half_height)
        bottom_right = (cx + half_width, cy + half_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Distancia pulgar-indice escala el rectangulo",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Presiona 'q' para salir",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("MediaPipe Hand Rectangle Scaling", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(f"Usando entorno en: {root}")
    main()
