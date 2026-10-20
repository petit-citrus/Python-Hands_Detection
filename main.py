"""Point d'entrée principal de l'application (Capture vidéo et Interface OpenCV)."""

import cv2
from hand_analyzer import HandAnalyzer
from hand_analyzer_mediapipe import MediaPipeHandAnalyzer


def main():
    # Ouvrir la webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    # Configurer la résolution souhaitée
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Créer la fenêtre principale
    nom_fenetre = "TP4 - Detection des mains (OpenCV)"
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
    nom_fenetre_mediapipe = "Detection des mains (MediaPipe)"
    cv2.namedWindow(nom_fenetre_mediapipe, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nom_fenetre, 800, 600)
    cv2.resizeWindow(nom_fenetre_mediapipe, 800, 600)
    cv2.moveWindow(nom_fenetre, 0, 0)
    cv2.moveWindow(nom_fenetre_mediapipe, 810, 0)

    # Initialiser l'analyseur géométrique OpenCV
    analyzer = HandAnalyzer()
    mediapipe_analyzer = MediaPipeHandAnalyzer()

    try:
        while True:
            # Capturer une image
            succes, image = camera.read()

            if not succes:
                print("Erreur : capture impossible.")
                break

            # Effet miroir
            image = cv2.flip(image, 1)

            mediapipe_image, mediapipe_hands = mediapipe_analyzer.process_frame(
                image.copy()
            )

            # Détecter la main et tracer le squelette dans la zone ROI
            image, hand_data = analyzer.process_frame(image)

            # Afficher le titre
            cv2.putText(
                image,
                "Detection des mains",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Afficher l'instruction
            cv2.putText(
                image,
                "Appuyer sur Q pour quitter",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Afficher la vidéo principale et la fenêtre du masque binaire (très apprécié en TP)
            cv2.imshow(nom_fenetre, image)
            cv2.imshow(nom_fenetre_mediapipe, mediapipe_image)
            cv2.imshow("Masque Binaire (YCrCb)", hand_data["mask"])

            # Quitter avec Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Libérer les ressources
        camera.release()
        mediapipe_analyzer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()