import cv2
import numpy as np
import matplotlib.pyplot as plt

class PrescriptionPreprocessor:
    def __init__(self, block_size=11, c=2):
        """
        Initialise le préprocesseur.
        :param block_size: Taille du voisinage pour la binarisation adaptative (doit être impair).
        :param c: Constante soustraite de la moyenne calculée (pour ajuster le contraste).
        """
        self.block_size = block_size
        self.c = c

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image à : {image_path}")
        return image

    def process(self, image_path):
        # 1. Chargement et passage en niveaux de gris
        img = self.load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Débruitage (Filtre Bilatéral)
        # Conserve les contours nets (le texte) tout en lissant le fond (le papier)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Binarisation Adaptative (Seuillage)
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C gère très bien les ombres et l'éclairage inégal
        binary = cv2.adaptiveThreshold(
            denoised,
            255, # Valeur max (blanc)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )

        # 4. Opérations morphologiques (Optionnel, à tester selon ton dataset)
        # Utile si l'encre est fine et se casse lors de la binarisation
        kernel = np.ones((2,2), np.uint8)
        # Une érosion sur fond blanc = épaissir le texte noir
        processed = cv2.erode(binary, kernel, iterations=1) 

        return img, processed

    def visualize(self, original, processed):
        """Affiche l'image avant et après pour comparer."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Image Originale")
        # Matplotlib utilise RGB, OpenCV utilise BGR
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Après Prétraitement")
        plt.imshow(processed, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# --- Exemple d'utilisation ---
# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Instancier la classe avec les paramètres par défaut
    preprocessor = PrescriptionPreprocessor(block_size=85, c=12)
    
    # 1. Remplace le chemin ci-dessous par le chemin réel de ton image
    chemin_image = "/home/Abdelali/Pictures/med/10.jpg" 
    
    # 2. On exécute le traitement
    img_orig, img_proc = preprocessor.process(chemin_image)
    
    # 3. On affiche le résultat avant/après
    preprocessor.visualize(img_orig, img_proc)