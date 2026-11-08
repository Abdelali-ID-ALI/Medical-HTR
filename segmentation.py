# src/preprocessing/segmentation.py

import cv2
import numpy as np
import os
from pathlib import Path


class LineSegmentor:
    """
    Segmente une ordonnance médicale en lignes individuelles.
    Chaque ligne est ensuite passée à TrOCR pour la reconnaissance.
    """

    def __init__(
        self,
        kernel_width: int = 50,
        kernel_height: int = 2,
        min_line_width: int = 100,
        min_line_height: int = 20,
        padding: int = 5,
        trocr_target_height: int = 64,
        quality_threshold: float = 0.1,
    ):
        """
        Args:
            kernel_width      : largeur du noyau morphologique (fusionne les mots d'une ligne)
            kernel_height     : hauteur du noyau (ne touche pas les lignes voisines)
            min_line_width    : largeur minimale pour qu'un contour soit une ligne valide
            min_line_height   : hauteur minimale pour qu'un contour soit une ligne valide
            padding           : marge ajoutée autour de chaque ligne découpée
            trocr_target_height: hauteur cible pour TrOCR (64px par défaut)
            quality_threshold : score minimum pour qu'une ligne soit conservée
        """
        self.kernel_width       = kernel_width
        self.kernel_height      = kernel_height
        self.min_line_width     = min_line_width
        self.min_line_height    = min_line_height
        self.padding            = padding
        self.trocr_target_height = trocr_target_height
        self.quality_threshold  = quality_threshold

    # ------------------------------------------------------------------ #
    #  ÉTAPE 0 — Correction de l'inclinaison                              #
    # ------------------------------------------------------------------ #

    def correct_skew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Détecte et corrige l'inclinaison de l'image (skew correction).
        Les photos de téléphone sont rarement parfaitement droites.

        Returns:
            (image corrigée, angle détecté en degrés)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image.copy()

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return image, 0.0

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = (theta - np.pi / 2) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = float(np.median(angles))

        # Inclinaison négligeable — pas besoin de corriger
        if abs(median_angle) < 0.5:
            return image, median_angle

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        print(f"  Inclinaison corrigée : {median_angle:.2f}°")
        return corrected, median_angle

    # ------------------------------------------------------------------ #
    #  ÉTAPE 1 — Binarisation interne                                     #
    # ------------------------------------------------------------------ #

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Binarisation adaptative avec flou gaussien préalable.
        THRESH_BINARY_INV → texte blanc sur fond noir (requis par dilate).
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary_inv = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,   # taille du bloc local
            15    # constante de soustraction
        )
        return binary_inv

    # ------------------------------------------------------------------ #
    #  ÉTAPE 2 — Dilatation horizontale pour fusionner les mots           #
    # ------------------------------------------------------------------ #

    def _dilate_lines(self, binary_inv: np.ndarray) -> np.ndarray:
        """
        Fusionne les mots d'une même ligne sans toucher les lignes voisines.
        kernel_width  large → fusionne les mots
        kernel_height fin   → respecte l'espacement inter-lignes
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_width, self.kernel_height)
        )
        dilated = cv2.dilate(binary_inv, kernel, iterations=1)
        return dilated

    # ------------------------------------------------------------------ #
    #  ÉTAPE 3 — Détection des contours de lignes                        #
    # ------------------------------------------------------------------ #

    def _find_line_contours(
        self, dilated: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Trouve les bounding boxes de chaque ligne, triées de haut en bas.

        Returns:
            Liste de tuples (x, y, w, h) triés verticalement
        """
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.min_line_width and h > self.min_line_height:
                boxes.append((x, y, w, h))

        # Tri vertical — lire l'ordonnance dans le bon ordre
        boxes.sort(key=lambda b: b[1])
        return boxes

    # ------------------------------------------------------------------ #
    #  ÉTAPE 4 — Découpage des lignes avec padding                       #
    # ------------------------------------------------------------------ #

    def _crop_line(
        self,
        image_original: np.ndarray,
        box: tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Découpe une ligne de l'image originale (couleur, non binarisée)
        avec une marge de sécurité.
        """
        x, y, w, h = box
        img_h, img_w = image_original.shape[:2]

        y1 = max(0, y - self.padding)
        y2 = min(img_h, y + h + self.padding)
        x1 = max(0, x - self.padding)
        x2 = min(img_w, x + w + self.padding)

        return image_original[y1:y2, x1:x2]

    # ------------------------------------------------------------------ #
    #  ÉTAPE 5 — Préparation pour TrOCR                                  #
    # ------------------------------------------------------------------ #

    def prepare_for_trocr(self, line_image: np.ndarray) -> np.ndarray:
        """
        Redimensionne la ligne à la hauteur cible de TrOCR
        en conservant les proportions (interpolation Lanczos).

        TrOCR attend des images de hauteur ~64px.
        """
        h, w = line_image.shape[:2]

        if h == 0 or w == 0:
            return line_image

        ratio   = self.trocr_target_height / h
        new_w   = max(1, int(w * ratio))

        resized = cv2.resize(
            line_image,
            (new_w, self.trocr_target_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        return resized

    # ------------------------------------------------------------------ #
    #  ÉTAPE 6 — Score de qualité par ligne                              #
    # ------------------------------------------------------------------ #

    def quality_score(self, line_image: np.ndarray) -> float:
        """
        Calcule un score de qualité (0 à 1) pour une ligne extraite.

        Combine :
        - ratio de pixels sombres (présence de texte)
        - variance du Laplacien (netteté de l'image)

        Score < quality_threshold → ligne ignorée (vide ou floue)
        """
        if line_image is None or line_image.size == 0:
            return 0.0

        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) \
            if len(line_image.shape) == 3 else line_image

        # Ratio de pixels sombres = présence de texte
        dark_ratio = np.sum(gray < 128) / gray.size

        # Variance Laplacien = mesure de netteté
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        score = min(1.0, dark_ratio * 10) * min(1.0, laplacian_var / 100)
        return round(float(score), 3)

    # ------------------------------------------------------------------ #
    #  MÉTHODE PRINCIPALE                                                 #
    # ------------------------------------------------------------------ #

    def segment(
        self,
        image_path: str,
        correct_skew: bool = True,
        prepare_trocr: bool = True,
    ) -> list[dict]:
        """
        Pipeline complet de segmentation d'une ordonnance en lignes.

        Args:
            image_path    : chemin vers l'image d'ordonnance
            correct_skew  : appliquer la correction d'inclinaison
            prepare_trocr : redimensionner pour TrOCR

        Returns:
            Liste de dicts :
            {
                "index"        : int,
                "image"        : np.ndarray (ligne originale),
                "image_trocr"  : np.ndarray (ligne redimensionnée),
                "bbox"         : (x, y, w, h),
                "quality_score": float,
            }
        """
        # Chargement
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")

        image_orig = image.copy()

        # 0 — Correction inclinaison
        skew_angle = 0.0
        if correct_skew:
            image, skew_angle = self.correct_skew(image)
            image_orig = image.copy()

        # 1 — Passage en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2 — Binarisation (BINARY_INV pour la dilatation)
        binary_inv = self._binarize(gray)

        # 3 — Dilatation horizontale
        dilated = self._dilate_lines(binary_inv)

        # 4 — Détection des contours
        boxes = self._find_line_contours(dilated)

        print(f"  {len(boxes)} lignes détectées"
              f"{f' (inclinaison : {skew_angle:.1f}°)' if correct_skew else ''}")

        # 5 — Découpage + qualité
        lignes = []
        for i, box in enumerate(boxes):
            crop = self._crop_line(image_orig, box)

            score = self.quality_score(crop)
            if score < self.quality_threshold:
                print(f"  Ligne {i} ignorée (qualité : {score})")
                continue

            trocr_ready = self.prepare_for_trocr(crop) if prepare_trocr else crop

            lignes.append({
                "index":         len(lignes),
                "image":         crop,
                "image_trocr":   trocr_ready,
                "bbox":          box,
                "quality_score": score,
            })

        print(f"  {len(lignes)} lignes valides retenues "
              f"(seuil qualité : {self.quality_threshold})")

        return lignes

    # ------------------------------------------------------------------ #
    #  SAUVEGARDE                                                         #
    # ------------------------------------------------------------------ #

    def save_lines(
        self,
        lignes: list[dict],
        output_dir: str,
        use_trocr_version: bool = True
    ) -> list[str]:
        """
        Sauvegarde les lignes découpées dans un dossier.

        Args:
            lignes           : résultat de segment()
            output_dir       : dossier de sortie
            use_trocr_version: sauvegarder la version redimensionnée pour TrOCR

        Returns:
            Liste des chemins sauvegardés
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for ligne in lignes:
            img_to_save = ligne["image_trocr"] if use_trocr_version \
                else ligne["image"]

            filename = f"ligne_{str(ligne['index']).zfill(2)}.jpg"
            path     = os.path.join(output_dir, filename)

            cv2.imwrite(path, img_to_save)
            saved_paths.append(path)

        print(f"  {len(saved_paths)} lignes sauvegardées dans '{output_dir}'")
        return saved_paths

    # ------------------------------------------------------------------ #
    #  VISUALISATION DEBUG                                                #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image_path: str,
        lignes: list[dict],
        scale: float = 0.5
    ) -> None:
        """
        Affiche l'image avec les bounding boxes des lignes détectées.
        Couleur = score de qualité (vert = bon, orange = moyen, rouge = mauvais)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossible de charger : {image_path}")
            return

        for ligne in lignes:
            x, y, w, h = ligne["bbox"]
            score       = ligne["quality_score"]

            # Couleur selon la qualité
            if score > 0.5:
                color = (0, 200, 0)    # vert
            elif score > 0.2:
                color = (0, 165, 255)  # orange
            else:
                color = (0, 0, 220)    # rouge

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                f"L{ligne['index']} q={score}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1
            )

        resized = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale))
        )

        cv2.imshow("Segmentation des lignes", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ #
#  POINT D'ENTRÉE                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    image_path  = sys.argv[1] if len(sys.argv) > 1 \
        else "/home/Abdelali/Pictures/med/10.jpg"
    output_dir  = sys.argv[2] if len(sys.argv) > 2 \
        else "/home/Abdelali/Pictures/med/lignes"

    segmentor = LineSegmentor(
        kernel_width=50,
        kernel_height=2,
        min_line_width=100,
        min_line_height=20,
        padding=5,
        trocr_target_height=64,
        quality_threshold=0.1,
    )

    print(f"Segmentation de : {image_path}")
    lignes = segmentor.segment(image_path, correct_skew=True)

    segmentor.save_lines(lignes, output_dir)
    segmentor.visualize(image_path, lignes)

    print("\nRésumé :")
    for l in lignes:
        print(f"  Ligne {l['index']} | bbox={l['bbox']} "
              f"| qualité={l['quality_score']} "
              f"| taille={l['image_trocr'].shape}")