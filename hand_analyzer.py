"""Détection et traçage du squelette de la main avec MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np

class HandAnalyzer:
    """Détecte les mains et extrait les 21 points clés."""

    def __init__(self, max_num_hands: int = 2, min_confidence: float = 0.7):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._drawing_utils = mp.solutions.drawing_utils
        self._hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._hands.process(rgb_frame)
        hands_data = []

        if not results.multi_hand_landmarks:
            return annotated_frame, hands_data

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            handedness = handedness_info.classification[0].label

            landmarks = [
                (int(point.x * width), int(point.y * height))
                for point in hand_landmarks.landmark
            ]

            center = self._get_palm_center(landmarks)

            self._drawing_utils.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self._hand_connections,
                self._drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self._drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            cv2.circle(annotated_frame, center, 5, (255, 0, 0), -1)

            hands_data.append({
                "handedness": handedness,
                "landmarks": landmarks,
                "center": center
            })

        return annotated_frame, hands_data

    @staticmethod
    def _get_palm_center(landmarks: list[tuple[int, int]]) -> tuple[int, int]:
        palm_indices = (0, 5, 9, 13, 17)
        return (
            int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices)),
            int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices)),
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with HandAnalyzer() as analyzer:
        capture = cv2.VideoCapture(0)

        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated_frame, hands_data = analyzer.process_frame(frame)

            for hand in hands_data:
                cv2.putText(
                    annotated_frame,
                    hand["handedness"],
                    (hand["center"][0] - 50, hand["center"][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

            cv2.imshow("Hand Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()