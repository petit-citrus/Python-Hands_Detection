"""Détection et traçage des 21 points de la main avec MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeHandAnalyzer:
    """Détecte les mains et trace leurs 21 points clés."""

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
            # Les coordonnées normalisées permettent de comparer les doigts indépendamment de la résolution de la webcam.
            finger_count = self._count_fingers(hand_landmarks.landmark, handedness)
            landmarks = [
                (
                    min(width - 1, max(0, int(point.x * width))),
                    min(height - 1, max(0, int(point.y * height))),
                )
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
            cv2.putText(
                annotated_frame,
                f"{handedness}: {finger_count} doigt(s)",
                (center[0] - 80, max(25, center[1] - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            hands_data.append(
                {
                    "handedness": handedness,
                    "landmarks": landmarks,
                    "center": center,
                    "finger_count": finger_count,
                }
            )

        return annotated_frame, hands_data

    @staticmethod
    def _count_fingers(landmarks, handedness: str) -> int:
        """Compte les doigts ouverts à partir des positions des articulations.

        Pour les quatre doigts verticaux, un doigt est ouvert lorsque son
        extrémité (TIP) est au-dessus de son articulation PIP. Le pouce est
        traité séparément sur l'axe horizontal car il s'écarte sur le côté.
        Les indices utilisés sont ceux du modèle MediaPipe Hands : 4/3 pour le
        pouce et 8/6, 12/10, 16/14, 20/18 pour les autres doigts.
        """
        count = 0

        # MediaPipe inverse généralement la latéralité sur une image miroir :
        # ce test est donc basé sur le label fourni avec les landmarks.
        if handedness == "Right":
            if landmarks[4].x < landmarks[3].x:
                count += 1
        elif landmarks[4].x > landmarks[3].x:
            count += 1

        # Le repère image a son origine en haut à gauche : y plus petit = plus haut.
        for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if landmarks[tip_index].y < landmarks[pip_index].y:
                count += 1

        return count

    @staticmethod
    def _get_palm_center(landmarks: list[tuple[int, int]]) -> tuple[int, int]:
        palm_indices = (0, 5, 9, 13, 17)
        return (
            int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices)),
            int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices)),
        )

    def close(self) -> None:
        self._hands.close()
