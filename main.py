import cv2
from hand_analyzer import HandAnalyzer


def main():
    # Ouvrir la webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    # Configurer la résolution souhaitée
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Créer la fenêtre
    nom_fenetre = "TP4 - Detection des mains"
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)

    # Initialiser le module de détection des mains
    with HandAnalyzer() as analyzer:
        try:
            while True:
                # Capturer une image
                succes, image = camera.read()

                if not succes:
                    print("Erreur : capture impossible.")
                    break

                # Effet miroir
                image = cv2.flip(image, 1)

                # Détecter les mains et tracer le squelette
                image, hands_data = analyzer.process_frame(image)

                # Afficher la latéralité (Left / Right) sur chaque main détectée
                for hand in hands_data:
                    cv2.putText(
                        image,
                        hand["handedness"],
                        (hand["center"][0] - 40, hand["center"][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

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

                # Afficher la vidéo
                cv2.imshow(nom_fenetre, image)

                # Quitter avec Q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            # Libérer les ressources
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()