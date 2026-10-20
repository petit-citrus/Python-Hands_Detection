"""Module de détection de la main et du squelette par méthodes classiques (100% OpenCV).

Méthodologie :
1. Découpage d'une Région d'Intérêt (ROI) pour isoler la zone de travail.
2. Segmentation couleur dans l'espace YCrCb (robuste aux variations de lumière).
3. Nettoyage morphologique (flou gaussien + fermeture/ouverture).
4. Extraction du plus grand contour (la main) et calcul des moments (centre de la paume).
5. Calcul de l'Enveloppe Convexe (Convex Hull) et des Défauts de Convexité (creux inter-doigts).
6. Filtrage géométrique par la loi des cosinus (angle < 90°) et traçage du squelette.
"""

import math
import cv2
import numpy as np


class HandAnalyzer:
    """Détecte la main, trace son squelette et extrait les doigts."""

    def __init__(
        self,
        roi_box: tuple[int, int, int, int] = (0, 0, 640, 480),
        ycrcb_min: tuple[int, int, int] = (0, 133, 77),
        ycrcb_max: tuple[int, int, int] = (255, 173, 127),
        min_contour_area: int = 3000,
    ):
        """Initialise les paramètres de traitement d'image.

        Args:
            roi_box: Coordonnées (x1, y1, x2, y2) du cadre dans lequel placer la main.
            ycrcb_min: Seuils bas (Y, Cr, Cb) pour la teinte de peau humaine.
            ycrcb_max: Seuils hauts (Y, Cr, Cb) pour la teinte de peau humaine.
            min_contour_area: Surface minimale (en pixels) pour considérer qu'une main est présente.
        """
        self.roi_box = roi_box
        self.ycrcb_min = np.array(ycrcb_min, dtype=np.uint8)
        self.ycrcb_max = np.array(ycrcb_max, dtype=np.uint8)
        self.min_contour_area = min_contour_area
        # Élément structurant (matrice 5x5) pour les opérations morphologiques
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Analyse l'image, trace le squelette géométrique et renvoie les données.

        Returns:
            Un tuple (annotated_frame, hand_data) contenant l'image annotée et un
            dictionnaire avec le masque binaire, le centre, les bouts de doigts, etc.
        """
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = self.roi_box

        # Dessiner le cadre de la zone de détection (ROI) à l'écran
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            "Placez la main ici",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Extraction de la sous-image (ROI)
        roi = frame[y1:y2, x1:x2]

        # Flou Gaussien pour réduire le bruit du capteur vidéo
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Conversion BGR -> YCrCb et seuillage (Thresholding) de la peau
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.ycrcb_min, self.ycrcb_max)

        # Morphologie mathématique pour nettoyer le masque binaire
        # CLOSE (dilatation puis érosion) : bouche les trous noirs dans la main
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # OPEN (érosion puis dilatation) : supprime les points blancs isolés du fond
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # Initialisation des données de sortie par défaut
        hand_data = {
            "detected": False,
            "center": None,
            "fingertips": [],
            "finger_count": 0,
            "mask": mask,
        }

        # Extraction des contours (algorithme de Suzuki & Abe dans OpenCV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return annotated_frame, hand_data

        # On sélectionne le contour ayant la plus grande superficie
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < self.min_contour_area:
            return annotated_frame, hand_data

        hand_data["detected"] = True

        # Décalage des coordonnées du contour pour les replacer dans l'image globale
        contour_global = max_contour + np.array([[x1, y1]])

        # Calcul du centre de masse de la main (Moments géométriques d'ordre 0 et 1)
        moments = cv2.moments(contour_global)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        center = (cx, cy)
        hand_data["center"] = center

        # Enveloppe convexe (Convex Hull)
        # returnPoints=True renvoie les coordonnées pour dessiner le polygone
        hull_points = cv2.convexHull(contour_global, returnPoints=True)
        # returnPoints=False renvoie les indices pour calculer les creux (défauts)
        hull_indices = cv2.convexHull(contour_global, returnPoints=False)

        # Dessin du contour de la main (gris) et de l'enveloppe convexe (vert)
        cv2.drawContours(annotated_frame, [contour_global], -1, (180, 180, 180), 1)
        cv2.drawContours(annotated_frame, [hull_points], -1, (0, 255, 0), 2)

        # ÉTAPE 6 : Défauts de convexité et filtrage géométrique des doigts
        fingertips = []
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour_global, hull_indices)
            if defects is not None:
                for s, e, f, depth in defects.reshape(-1, 4):
                    start = tuple(contour_global[s][0])  # Bout d'un doigt (début)
                    end = tuple(contour_global[e][0])    # Bout du doigt suivant (fin)
                    far = tuple(contour_global[f][0])    # Creux entre les deux doigts

                    # Longueurs des 3 côtés du triangle formé par (start, end, far)
                    a = math.hypot(end[0] - start[0], end[1] - start[1])
                    b = math.hypot(far[0] - start[0], far[1] - start[1])
                    c = math.hypot(end[0] - far[0], end[1] - far[1])

                    if b == 0 or c == 0:
                        continue

                    # Loi des cosinus : calcul de l'angle au creux entre deux doigts
                    cos_angle = max(-1.0, min(1.0, (b**2 + c**2 - a**2) / (2 * b * c)))
                    angle = math.degrees(math.acos(cos_angle))

                    # depth est codé en virgule fixe par OpenCV (diviser par 256 pour avoir des pixels)
                    real_depth = depth / 256.0

                    # Si l'angle est aigu (< 90°) et le creux assez profond (> 20 px),
                    # alors ce sont bien deux doigts écartés
                    if angle <= 90 and real_depth > 20:
                        # Point bleu au fond du creux inter-doigt
                        cv2.circle(annotated_frame, far, 5, (255, 0, 0), -1)
                        self._add_unique_point(fingertips, start)
                        self._add_unique_point(fingertips, end)

        # Cas particulier : si un seul doigt est levé (aucun creux entre 2 doigts),
        # on prend le point le plus haut du contour s'il est bien au-dessus de la paume
        if not fingertips:
            topmost = tuple(contour_global[contour_global[:, :, 1].argmin()][0])
            if center[1] - topmost[1] > 60:
                fingertips.append(topmost)

        # ÉTAPE 7 : Traçage du squelette (Lignes reliant le centre aux bouts des doigts)
        for tip in fingertips:
            # Ligne d'os (jaune/orange) entre le centre de la paume et le bout du doigt
            cv2.line(annotated_frame, center, tip, (0, 200, 255), 2)
            # Cercle rouge au bout du doigt détecté
            cv2.circle(annotated_frame, tip, 7, (0, 0, 255), -1)

        # Cercle central de la paume (magenta)
        cv2.circle(annotated_frame, center, 8, (255, 0, 255), -1)

        hand_data["fingertips"] = fingertips
        # La géométrie ne connaît pas les articulations : elle compte donc les
        # pointes validées par les défauts de convexité. C'est une estimation.
        hand_data["finger_count"] = min(len(fingertips), 5)
        cv2.putText(
            annotated_frame,
            f"Doigts (geometrique): {hand_data['finger_count']}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )
        return annotated_frame, hand_data

    @staticmethod
    def _add_unique_point(
        points_list: list[tuple[int, int]],
        new_point: tuple[int, int],
        min_dist: int = 25,
    ) -> None:
        """Évite d'ajouter deux fois le même bout de doigt s'ils sont très proches."""
        for pt in points_list:
            if math.hypot(pt[0] - new_point[0], pt[1] - new_point[1]) < min_dist:
                return
        points_list.append(new_point)